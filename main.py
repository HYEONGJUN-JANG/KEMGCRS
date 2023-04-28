import copy
import json
import sys
import logging
from collections import defaultdict
import random

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BartForConditionalGeneration, GPT2LMHeadModel, GPT2Config
import data
from config import bert_special_tokens_dict, gpt_special_tokens_dict
from data_model import GenerationDataset, DialogDataset, KnowledgeDataset
from eval_know import eval_know, knowledge_reindexing
from train_know import train_know
from utils import *
from models import *
from data_util import readDic, dataset_reader, process_augment_sample
from train_goal_topic import topic_eval


# def train_knowledge_indexing(args, knowledge_data, retriever, optimizer):
#     # 모든 know_index를 버트에 태움
#     print('...train knowledge indexing...')
#     knowledgeDataLoader = DataLoader(
#         knowledge_data,
#         batch_size=args.batch_size
#     )
#     knowledge_index = []
#     criterion = nn.CrossEntropyLoss()
#     train_epoch_loss = 0
#     for batch in tqdm(knowledgeDataLoader):
#         input_ids = batch[0].to(args.device)
#         attention_mask = batch[1].to(args.device)
#         target_know_idx = batch[2].to(args.device)
#
#         if args.know_ablation == 'bart':
#             loss = retriever.knowledge_retrieve(input_ids, attention_mask, None, None, labels=target_know_idx)
#         else:
#             logit = retriever.knowledge_retrieve(input_ids, attention_mask, None, None, ablation='mlp')
#             loss = criterion(logit, target_know_idx)
#
#         train_epoch_loss += loss
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print(f"Knowledge Indexing Loss: {train_epoch_loss}")

def split_validation(train_dataset_raw, idx):
    sidx = np.arange(len(train_dataset_raw))
    # np.random.shuffle(sidx)
    n_train = int(np.round(len(train_dataset_raw) * (1. - 0.1)))
    valid_dataset = train_dataset_raw[n_train:]
    train_dataset = train_dataset_raw[:n_train]
    return train_dataset, valid_dataset


def main():
    # TH 작업 main
    args = parseargs()
    # args.data_cache = False
    args.who = "TH"
    # args.bert_name = 'facebook/bart-base'
    # args.task = 'know'
    args.max_gen_length = 50
    print(args)

    checkPath(args.log_dir)
    checkPath(args.model_dir)

    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(args.log_dir, f'{args.time}_{args.log_name + "_"}log.txt'), filemode='a', format='%(asctime)s: %(levelname)s: %(message)s', datefmt='%Y/%m/%d_%p_%I:%M:%S ')
    logging.info('Commend: {}'.format(', '.join(map(str, sys.argv))))
    # Model cached load
    checkPath(os.path.join("cache", args.bert_name))

    topicDic = readDic(os.path.join(args.data_dir, "topic2id.txt"))
    goalDic = readDic(os.path.join(args.data_dir, "goal2id.txt"))
    args.topicDic = topicDic
    args.goalDic = goalDic
    args.topic_num = len(topicDic['int'])
    args.goal_num = len(goalDic['int'])

    # Read knowledge DB
    # train_knowledgeDB = data.read_pkl(os.path.join(args.data_dir, 'train_knowledge_DB.pickle'))  # TODO: verbalize (TH)
    knowledgeDB = data.read_pkl(os.path.join(args.data_dir, 'knowledgeDB.txt'))  # TODO: verbalize (TH)
    knowledgeDB.insert(0, "")
    args.knowledge_num = len(knowledgeDB)
    args.knowledgeDB = knowledgeDB

    train_dataset_raw = dataset_reader(args, 'train')
    test_dataset_raw = dataset_reader(args, 'test')
    valid_dataset_raw = dataset_reader(args, 'dev')

    if 'resp' in args.task:

        # config = GPT2Config.from_pretrained(args.bert_name, max_length=args.max_gen_length+args.max_length)
        gpt_model = GPT2LMHeadModel.from_pretrained(args.gpt_name, cache_dir=os.path.join("cache", args.gpt_name))
        tokenizer = AutoTokenizer.from_pretrained(args.gpt_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens(gpt_special_tokens_dict)  # [TH] add bert special token (<dialog>, <topic> , <type>)

        gpt_model.resize_token_embeddings(len(tokenizer))
        args.hidden_size = gpt_model.config.hidden_size  # BERT large 쓸 때 대비

        train_dataset_resp = process_augment_sample(train_dataset_raw, tokenizer, knowledgeDB)
        test_dataset_resp = process_augment_sample(test_dataset_raw, tokenizer, knowledgeDB)

        train_datamodel_resp = GenerationDataset(args, train_dataset_resp, knowledgeDB, tokenizer, mode='train', knowledge=args.knowledge)
        test_datamodel_resp = GenerationDataset(args, test_dataset_resp, knowledgeDB, tokenizer, mode='test', knowledge=args.knowledge)

        train_dataloader_resp = DataLoader(train_datamodel_resp, batch_size=args.batch_size, shuffle=True)
        test_dataloader_resp = DataLoader(test_datamodel_resp, batch_size=args.batch_size, shuffle=False)

        generator = Retriever(args, gpt_model=gpt_model)
        generator = generator.to(args.device)
        criterion = nn.CrossEntropyLoss().to(args.device)
        optimizer = optim.AdamW(generator.parameters(), lr=args.lr)

        # train generate task
        if args.saved_model_path == '':
            for epoch in range(args.num_epochs):
                train_epoch_loss = 0
                for batch in tqdm(train_dataloader_resp, desc="Generate_Train", bar_format=' {l_bar} | {bar:23} {r_bar}'):
                    generator.train()
                    dialog_token = batch['input_ids'].to(args.device)
                    dialog_mask = batch['attention_mask'].to(args.device)
                    response = batch['response'].to(args.device)

                    loss = generator.generation(dialog_token, dialog_mask, response)
                    # loss = criterion(dot_score, targets)
                    train_epoch_loss += loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                print(f"Epoch: {epoch}\nTrain Loss: {train_epoch_loss}")
            torch.save(generator.state_dict(), os.path.join(args.model_dir, f"{args.time}_{args.model_name}_gen_bin.pt"))  # TIME_MODELNAME 형식

            # test generation task
            all_dialog = []
            all_response = []
            all_generated = []
            for batch in tqdm(test_dataloader_resp, desc="Generate Test", bar_format=' {l_bar} | {bar:23} {r_bar}'):
                generator.eval()
                dialog_token = batch['input_ids'].to(args.device)
                dialog_mask = batch['attention_mask'].to(args.device)
                response = batch['response']

                batch_size = dialog_token.shape[0]
                generated = generator.gpt_model.generate(input_ids=dialog_token,
                                                         attention_mask=dialog_mask,
                                                         pad_token_id=tokenizer.pad_token_id,
                                                         max_length=args.max_gen_length + args.max_length)
                # decoded_generated = tokenizer.batch_decode(generated)

                gen_resp_ids = []
                for gen_seq, length in zip(generated, batch['context_len']):
                    gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
                    gen_resp_ids.append(gen_seq[length:])

                all_generated.extend(tokenizer.batch_decode(gen_resp_ids))
                all_response.extend(response)
                all_dialog.extend(tokenizer.batch_decode(dialog_token, skip_special_tokens=True))

            with open(f"response_write_{args.time}_{args.model_name}.txt", 'w', encoding='UTF-8') as f:
                for (a, b, c) in zip(all_dialog, all_response, all_generated):
                    f.write('[DIALOG]\t%s\n[RESPONSE]\t%s\n[GENERATED]\t%s\n' % (a, b, c))
                    f.write('-------------------------------------------\n')
        else:
            generator.load_state_dict(torch.load(os.path.join(args.model_dir, args.saved_model_path)))

    if 'know' in args.task:
        # KNOWLEDGE TASk
        bert_model = AutoModel.from_pretrained(args.bert_name, cache_dir=os.path.join("cache", args.bert_name))
        tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
        tokenizer.add_special_tokens(bert_special_tokens_dict)  # [TH] add bert special token (<dialog>, <topic> , <type>)
        bert_model.resize_token_embeddings(len(tokenizer))
        args.hidden_size = bert_model.config.hidden_size  # BERT large 쓸 때 대비

        retriever = Retriever(args, bert_model)
        retriever = retriever.to(args.device)

        knowledge_data = KnowledgeDataset(args, knowledgeDB, tokenizer)  # knowledge dataset class
        args.knowledge_num = len(knowledgeDB)
        args.knowledgeDB = knowledgeDB

        # train_dataset_raw = dataset_reader(args, 'train')
        # test_dataset_raw = dataset_reader(args, 'test')
        train_dataset = process_augment_sample(train_dataset_raw, tokenizer, knowledgeDB)
        # train_dataset, valid_dataset = split_validation(train_dataset)
        valid_dataset = process_augment_sample(valid_dataset_raw, tokenizer, knowledgeDB)
        test_dataset = process_augment_sample(test_dataset_raw, tokenizer, knowledgeDB)

        train_datamodel_know = DialogDataset(args, train_dataset, knowledgeDB, tokenizer, task='know')
        valid_datamodel_know = DialogDataset(args, valid_dataset, knowledgeDB, tokenizer, task='know')
        test_datamodel_know = DialogDataset(args, test_dataset, knowledgeDB, tokenizer, task='know')

        train_dataloader = DataLoader(train_datamodel_know, batch_size=args.batch_size, shuffle=True)
        valid_dataloader = DataLoader(test_datamodel_know, batch_size=args.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_datamodel_know, batch_size=1, shuffle=False)

        train_know(args, train_dataloader, valid_dataloader, retriever, knowledge_data, knowledgeDB, tokenizer)
        eval_know(args, test_dataloader, retriever, knowledge_data, knowledgeDB, tokenizer, write=True)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리


if __name__ == "__main__":
    main()
    # logging.info('Commend: {}'.format(', '.join(map(str, sys.argv))))
