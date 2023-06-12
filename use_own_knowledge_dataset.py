import logging
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

import faiss
import torch
from datasets import Features, Sequence, Value, load_dataset
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import data_pseudo

from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    HfArgumentParser,
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenizer,
)

from data_model import DialogDataset

logger = logging.getLogger(__name__)
# torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"


def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i: i + n]).strip() for i in range(0, len(text), n)]


def split_documents(documents: dict) -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents["title"], documents["text"]):
        if text is not None:
            for passage in split_text(text):
                titles.append(title if title is not None else "")
                texts.append(passage)
    return {"title": titles, "text": texts}


def embed(documents: dict, ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast) -> dict:
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(
        documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt"
    )["input_ids"]

    embeddings = ctx_encoder(input_ids.to(device=device), return_dict=True).pooler_output
    return {"embeddings": embeddings.detach().cpu().numpy()}


class RagDataset(Dataset):
    def __init__(self, args, augmented_raw_sample, tokenizer):
        super(Dataset, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.augmented_raw_sample = augmented_raw_sample

    def __getitem__(self, item):
        data = self.augmented_raw_sample[item]
        cbdicKeys = ['dialog', 'user_profile', 'response', 'type', 'topic', 'situation', 'target_knowledge', 'candidate_knowledges', 'candidate_confidences']
        dialog, user_profile, response, type, topic, situation, target_knowledge_idx, candidate_knowledges, candidate_confidences = [data[i] for i in cbdicKeys]

        # input_dict = self.tokenizer.prepare_seq2seq_batch(dialog, response, return_tensors="pt", )

        input_ids = self.tokenizer(dialog, max_length=args.max_length, padding='max_length', truncation=True)['input_ids']
        attention_mask = self.tokenizer(dialog, max_length=args.max_length, padding='max_length', truncation=True)['attention_mask']

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(response, max_length=args.max_length, padding='max_length', truncation=True)['input_ids']

        context_batch = defaultdict()

        context_batch['input_ids'] = input_ids
        context_batch['attention_mask'] = attention_mask
        context_batch['labels'] = labels

        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.args.device)
                # context_batch[k] = torch.as_tensor(v)
        return context_batch


    def __len__(self):
        return len(self.augmented_raw_sample)


def main(
        rag_example_args: "RagExampleArguments",
        processing_args: "ProcessingArguments",
        index_hnsw_args: "IndexHnswArguments",
):
    ######################################
    logger.info("Step 1 - Create the dataset")
    ######################################

    # The dataset needed for RAG must have three columns:
    # - title (string): title of the document
    # - text (string): text of a passage of the document
    # - embeddings (array of dimension d): DPR representation of the passage

    # Let's say you have documents in tab-separated csv files with columns "title" and "text"
    assert os.path.isfile(rag_example_args.csv_path), "Please provide a valid path to a csv file"

    # You can load a Dataset object this way
    dataset = load_dataset(
        "csv", data_files=[rag_example_args.csv_path], split="train", delimiter="\t", column_names=["title", "text"]
    )

    # More info about loading csv files in the documentation: https://huggingface.co/docs/datasets/loading_datasets.html?highlight=csv#csv-files

    # Then split the documents into passages of 100 words
    dataset = dataset.map(split_documents, batched=True, num_proc=processing_args.num_proc)

    # And compute the embeddings
    ctx_encoder = DPRContextEncoder.from_pretrained(rag_example_args.dpr_ctx_encoder_model_name).to(device=device)
    ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(rag_example_args.dpr_ctx_encoder_model_name)
    new_features = Features(
        {"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))}
    )  # optional, save as float32 instead of float64 to save space
    dataset = dataset.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer),
        batched=True,
        batch_size=processing_args.batch_size,
        features=new_features,
    )

    # And finally save your dataset
    passages_path = os.path.join(rag_example_args.output_dir, "my_knowledge_dataset")
    dataset.save_to_disk(passages_path)
    # from datasets import load_from_disk
    # dataset = load_from_disk(passages_path)  # to reload the dataset

    ######################################
    logger.info("Step 2 - Index the dataset")
    ######################################

    # Let's use the Faiss implementation of HNSW for fast approximate nearest neighbor search
    index = faiss.IndexHNSWFlat(index_hnsw_args.d, index_hnsw_args.m, faiss.METRIC_INNER_PRODUCT)
    dataset.add_faiss_index("embeddings", custom_index=index)

    # And save the index
    index_path = os.path.join(rag_example_args.output_dir, "my_knowledge_dataset_hnsw_index.faiss")
    dataset.get_index("embeddings").save(index_path)
    # dataset.load_faiss_index("embeddings", index_path)  # to reload the index

    ######################################
    logger.info("Step 3 - Load RAG")
    ######################################

    # Easy way to load the model
    # retriever = RagRetriever.from_pretrained(
    #     rag_example_args.rag_model_name, index_name="custom", indexed_dataset=dataset
    # )
    retriever = RagRetriever.from_pretrained('facebook/rag-sequence-nq', index_name='custom', indexed_dataset=dataset, init_retrieval=True)
    model = RagSequenceForGeneration.from_pretrained(rag_example_args.rag_model_name, retriever=retriever).to(args.device)
    tokenizer = RagTokenizer.from_pretrained(rag_example_args.rag_model_name)

    # For distributed fine-tuning you'll need to provide the paths instead, as the dataset and the index are loaded separately.
    # retriever = RagRetriever.from_pretrained(rag_model_name, index_name="custom", passages_path=passages_path, index_path=index_path)

    # ######################################
    # logger.info("Step 4 - Have fun")
    # ######################################
    #
    # question = rag_example_args.question or "What is the star sign of Xun Zhou?"
    # input_ids = tokenizer.question_encoder(question, return_tensors="pt")["input_ids"]
    #
    # # 1. Encode
    # question_hidden_states = model.question_encoder(input_ids)[0]
    # # 2. Retrieve
    # docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
    # doc_scores = torch.bmm(
    #     question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
    # ).squeeze(1)
    #
    # generated = model.generate(input_ids)
    # generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    # logger.info("Q: " + question)
    # logger.info("A: " + generated_string)

    ######################################
    logger.info("Step 5 - Fine-tuning")
    ######################################
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    train_datamodel_know = RagDataset(args, train_dataset_resp, tokenizer)
    train_dataloader = DataLoader(train_datamodel_know, batch_size=args.batch_size, shuffle=True)


    # bert_model = AutoModel.from_pretrained(args.bert_name).to(args.device)
    # bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_name)

    for epoch in range(10):
        train_epoch_loss = 0
        model.train()

        for batch in tqdm(train_dataloader, desc="Knowledge_Train", bar_format=' {l_bar} | {bar:23} {r_bar}'):
            dialog_token = batch['input_ids']
            dialog_mask = batch['attention_mask']
            response = batch['labels']
            outputs = model(input_ids=dialog_token, attention_mask=dialog_mask, labels=response)
            loss = outputs.loss.mean()
            train_epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("LOSS:\t%.4f" % train_epoch_loss)
        model.eval()

    with torch.no_grad():
        for batch in tqdm(train_dataloader, desc="Knowledge_Train", bar_format=' {l_bar} | {bar:23} {r_bar}'):
            dialog_token = batch['input_ids']
            dialog_mask = batch['attention_mask']
            response = batch['labels']

            # 1. Encode
            # input_ids = tokenizer.question_encoder(dialog_token, return_tensors="pt")["input_ids"]
            question_hidden_states = model.question_encoder(dialog_token, dialog_mask)[0]  # model.question_encoder(input_ids)[0]
            # 2. Retrieve
            # docs_dict = retriever(dialog_token.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
            docs_dict = retriever(dialog_token.cpu().numpy(), question_hidden_states.cpu().detach().numpy(), return_tensors="pt").to(args.device)

            doc_scores = torch.bmm(
                question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
            ).squeeze(1)

            generated = model.generate(
                context_input_ids=docs_dict["context_input_ids"],
                context_attention_mask=docs_dict["context_attention_mask"],
                doc_scores=doc_scores,
            )
            generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
            print("[dialog]\n%s" % tokenizer.question_encoder.batch_decode(dialog_token, skip_special_tokens=True))
            print('[generated]\n%s' % generated_string)
            print("[response]\n%s" % tokenizer.batch_decode(response, skip_special_tokens=True))

    # inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
    # targets = tokenizer(text_target="In Paris, there are 10 million people.", return_tensors="pt")
    # input_ids = inputs["input_ids"]
    # labels = targets["input_ids"]
    # outputs = model(input_ids=input_ids, labels=labels)


@dataclass
class RagExampleArguments:
    csv_path: str = field(
        default=str(Path(__file__).parent / "test_data" / "my_knowledge_dataset.csv"),
        metadata={"help": "Path to a tab-separated csv file with columns 'title' and 'text'"},
    )
    question: Optional[str] = field(
        default=None,
        metadata={"help": "Question that is passed as input to RAG. Default is 'What does Moses' rod turn into ?'."},
    )
    rag_model_name: str = field(
        default="facebook/rag-sequence-nq",
        metadata={"help": "The RAG model to use. Either 'facebook/rag-sequence-nq' or 'facebook/rag-token-nq'"},
    )
    dpr_ctx_encoder_model_name: str = field(
        default="facebook/dpr-ctx_encoder-multiset-base",
        metadata={
            "help": (
                "The DPR context encoder model to use. Either 'facebook/dpr-ctx_encoder-single-nq-base' or"
                " 'facebook/dpr-ctx_encoder-multiset-base'"
            )
        },
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a directory where the dataset passages and the index will be saved"},
    )


@dataclass
class ProcessingArguments:
    num_proc: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use to split the documents into passages. Default is single process."
        },
    )
    batch_size: int = field(
        default=16,
        metadata={
            "help": "The batch size to use when computing the passages embeddings using the DPR context encoder."
        },
    )


@dataclass
class IndexHnswArguments:
    d: int = field(
        default=768,
        metadata={"help": "The dimension of the embeddings to pass to the HNSW Faiss index."},
    )
    m: int = field(
        default=128,
        metadata={
            "help": (
                "The number of bi-directional links created for every new element during the HNSW index construction."
            )
        },
    )


import utils
import data_util
import os


def process_augment_sample(raw_data, tokenizer=None):
    train_sample = []
    if tokenizer:
        try:
            if tokenizer.eos_token is not None:
                eos_token = tokenizer.eos_token
            else:
                eos_token = tokenizer.sep_token
        except:
            eos_token = tokenizer.generator.eos_token
    else:
        eos_token = '</s>'
    for ij in tqdm(range(len(raw_data)), desc="Dataset Augment", bar_format='{l_bar} | {bar:23} {r_bar}'):
        conversation = raw_data[ij]
        augmented_dialog = []
        augmented_knowledge = []
        last_type = ""
        for i in range(len(conversation['dialog'])):
            role = conversation['role_seq'][i]
            utterance = conversation['dialog'][i] + eos_token
            goal = conversation['type'][i]
            if goal == 'Q&A':  # TH 230601
                if role == 'system' and len(augmented_dialog) > 0 and len(conversation['pseudo_knowledge_seq'][i]) != 0:
                    flatten_dialog = ''.join(augmented_dialog)
                    train_sample.append({'dialog': flatten_dialog,
                                         'user_profile': conversation['user_profile'],
                                         'response': utterance,
                                         'type': conversation['type'][i],
                                         'last_type': last_type,
                                         'topic': conversation['topic'][i],
                                         'situation': conversation['situation'],
                                         'related_knowledges': conversation['related_knowledges'],
                                         'augmented_knowledges': deepcopy(augmented_knowledge),  # TH related 대신에 know seq 230601
                                         'target_knowledge': conversation['knowledge_seq'][i],
                                         'candidate_knowledges': conversation['pseudo_knowledge_seq'][i],
                                         'candidate_confidences': conversation['pseudo_confidence_seq'][i]  # prob
                                         })
            if role == 'system': last_type = conversation['type'][i]
            augmented_dialog.append(utterance)
            augmented_knowledge.append(conversation['knowledge_seq'][i])
    return train_sample


if __name__ == "__main__":

    args = utils.parseargs()
    args.pseudo, args.usePseudoLabel, args.inputWithKnowledge = True, True, True
    train_dataset_raw, train_knowledge_seq_set = data_pseudo.dataset_reader(args, 'train')
    dev_dataset_raw, dev_knowledge_seq_set = data_pseudo.dataset_reader(args, 'dev')
    test_dataset_raw, test_knowledge_seq_set = data_pseudo.dataset_reader(args, 'test')

    test_knowledge_seq_set = train_knowledge_seq_set.union(dev_knowledge_seq_set).union(test_knowledge_seq_set)
    train_knowledge_seq_set = list(train_knowledge_seq_set)

    train_dataset_resp = process_augment_sample(train_dataset_raw)
    dev_dataset_resp = process_augment_sample(dev_dataset_raw)
    test_dataset_resp = process_augment_sample(test_dataset_raw)

    os.makedirs('test_data', exist_ok=True)


    def save(mode, listdataset):
        s_path, t_path = os.path.join('test_data', mode + '.source'), os.path.join(args.home, 'test_data', mode + '.target')
        with open(s_path, 'w', encoding='utf-8') as ss, open(t_path, 'w', encoding='utf-8') as tt:
            for data in listdataset:
                ss.write(f"{data['dialog']}\n")
                tt.write(f"{data['response']}\n")
                # data['response']
                # f.write(f"{know}\t{know}\n")


    # os.path.join(args.home, 'test_run','dummy-kb','my_knowledge_dataset.csv')
    with open(os.path.join('test_data', 'my_knowledge_dataset.csv'), 'w', encoding='utf-8') as f:
        for know in train_knowledge_seq_set:
            f.write(f" \t{know}\n")

    save('train', train_dataset_resp)
    save('val', dev_dataset_resp)
    save('test', test_dataset_resp)

    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)

    parser = HfArgumentParser((RagExampleArguments, ProcessingArguments, IndexHnswArguments))
    rag_example_args, processing_args, index_hnsw_args = parser.parse_args_into_dataclasses()
    with TemporaryDirectory() as tmp_dir:
        rag_example_args.output_dir = rag_example_args.output_dir or tmp_dir
        main(rag_example_args, processing_args, index_hnsw_args)
