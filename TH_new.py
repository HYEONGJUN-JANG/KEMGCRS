import json
import sys
import logging
from collections import defaultdict
import random

from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BartForConditionalGeneration
import data
from config import bert_special_tokens_dict
from eval_know import eval_know
from train_know import train_retriever_idx
from utils import *
from models import *
from data_util import readDic


def truncationPadding(input_ids, max_length, prefix=[], suffix=[]):
    truncate_size = max_length - len(prefix) - len(suffix)
    if truncate_size <= len(input_ids):
        input_ids = prefix + input_ids[len(input_ids) - truncate_size:] + suffix
    else:
        input_ids = prefix + input_ids + suffix
    return input_ids + [0] * (max_length - len(input_ids))


def user_profile_setting(ufDic: dict) -> str:
    uf = ''
    for i, key in enumerate(ufDic.keys()):
        one = ufDic[key]
        if i == 0 or key[0].lower() != "a":
            pass
        else:
            uf += ' | '
        if type(one) == list:
            uf += f"{key}: {', '.join(one[:-5])}"
        else:
            uf += f"{key}: {one}"
    return uf


def dataset_reader(args, tokenizer, knowledgeDB, data_name='train'):
    conversation_sample = []
    data_path = os.path.join(args.data_dir, f"en_{data_name}.txt")
    with open(data_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f, desc="Dataset Read", bar_format='{l_bar} | {bar:23} {r_bar}'):
            dialog = json.loads(line)
            conversation = dialog['conversation']

            role_seq = ["User", "System"] if dialog['goal_type_list'][0] != 'Greetings' else ["System", "User"]

            for i in range(2, len(conversation)):
                role_seq.append(role_seq[i % 2])

            knowledge_seq = dialog['knowledge']
            knowledge_seq = [' '.join(know) for know in knowledge_seq]
            user_profile = user_profile_setting(dialog['user_profile'])
            situation = dialog['situation']

            for i in range(len(conversation)):  # HJ: [1],[2] 같은 text 제거, conversation 추가해넣는코드
                conversation[i] = conversation[i] if conversation[i][0] != '[' else conversation[i][4:]
                conversation[i] = role_seq[i] + ": " + conversation[i]
            conversation_sample.append({
                'dialog': conversation,
                'role_seq': role_seq,
                'type': dialog['goal_type_list'],
                'topic': dialog['goal_topic_list'],
                'situation': situation,
                'user_profile': user_profile,
                'knowledge_seq': knowledge_seq
            })
    return conversation_sample


def negative_sampler(args, target_knowledge):
    # candidate_entity = self.knowledgeDB[target_knowledge][0]
    # candiate_all_list = self.knowledgeDB_entity_values[candidate_entity]
    # negative_indice = random.choices(candiate_all_list, k=self.args.negative_num if len(candiate_all_list) > self.args.negative_num else len(candiate_all_list))
    total_knowledge_num = args.knowledge_num
    negative_indice = []
    while len(negative_indice) < args.negative_num:
        negative_idx = random.randint(0, total_knowledge_num - 1)
        if (negative_idx not in negative_indice) and (negative_idx != target_knowledge):
            negative_indice.append(negative_idx)
    return negative_indice


def batchify(args, batch, knowledgeDB, tokenizer=None, task=''):
    """
    :param args: args
    :param batch: batch
    :param tokenizer: tokenizer
    :param task: task {'type','topic','know'}
    :return: Tensor[ dialog_token, dialog_mask, response, type, topic, candidate_indice(Optional) ]
    """
    # Input batches are all string
    dialog, user_profile, response, type, topic, situation, target_knowledge = [batch[i] for i in ['dialog', 'user_profile', 'response', 'type', 'topic', 'situation', 'target_knowledge']]
    context_batch = defaultdict()
    prefix_list = []
    for i in range(len(dialog)):  # batch 수 만큼
        prefix = ' '
        if task == 'type':
            prefix = tokenizer.sep_token
        elif task == 'topic':
            prefix = '<type>' + type[i] + '<user_profile>' + user_profile[i]
        elif task == 'know':
            if isinstance(topic[i], list): topic[i] = ','.join(topic[i])
            prefix = tokenizer.sep_token + '<situation>' + situation[i] + '<type>' + type[i] + '<topic>' + topic[i] + "predict the next goal:"
        else:  # Rescponse
            prefix = tokenizer.sep_token + '<knowledge>' + knowledgeDB[target_knowledge[i]] + "predict the next response:"
            pass
        prefix_list.append(prefix)

    input_sentences = [s + '<dialog>' + d for d, s in zip(dialog, prefix_list)]
    input_sentences = tokenizer(input_sentences, add_special_tokens=False).input_ids
    topic_prompt = tokenizer.encode('predict the next goal: ')[1:]
    input_sentences = [[tokenizer.cls_token_id] + sentence[-args.max_length + len(topic_prompt) + 1:] + topic_prompt for sentence in input_sentences]
    input_sentences = [input_ids + [tokenizer.pad_token_id] * (args.max_length - len(input_ids)) for input_ids in input_sentences]

    # suffix_list_token = tokenizer(suffix_list, add_special_tokens=False)
    # dialog_list_token = tokenizer(dialog, add_special_tokens=False)
    # input_token = [tokenizer.cls_token + s+d + " predict the next topic:" + tokenizer.eos_token for s, d in zip(suffix_list_token.input_ids, dialog_list_token.input_ids)]
    # input_encoding = tokenizer(input_token)
    context_batch['dialog_token'] = torch.LongTensor(input_sentences).to(args.device)
    attention_mask = context_batch['dialog_token'].ne(tokenizer.pad_token_id)
    context_batch['dialog_mask'] = attention_mask

    # tokenized_dialog = tokenizer(input_token, truncation=True, padding='max_length', max_length=args.max_length)
    # tokenized_dialog = tokenizer(dialog, add_special_tokens=False)
    # tokenized_suffix = tokenizer(suffix_list, add_special_tokens=False, max_length=args.max_length//4, truncation=True)
    # truncationPadding
    context_batch['response'] = tokenizer(response, add_special_tokens=True, max_length=args.max_length, padding='max_length', truncation=True).input_ids
    # context_batch['dialog_token'] = [truncationPadding(input_ids=dialog_inputids, prefix=[tokenizer.cls_token_id], suffix=suffix_inputids, max_length=args.max_length) for dialog_inputids, suffix_inputids in zip(tokenized_dialog.input_ids, tokenized_suffix.input_ids)]
    # context_batch['dialog_mask'] = [truncationPadding(input_ids=dialoginputids, prefix=[1], suffix=suffix_inputids, max_length=args.max_length) for dialoginputids, suffix_inputids in zip(tokenized_dialog.attention_mask, tokenized_suffix.attention_mask)]
    # context_batch['dialog_token'] = tokenized_dialog.input_ids
    # context_batch['dialog_mask'] = tokenized_dialog.attention_mask

    context_batch['type'] = [args.goalDic['str'][i] for i in type]  # index로 바꿈
    context_batch['topic_idx'] = [args.topicDic['str'][i] for i in topic]  # index로 바꿈
    context_batch['topic'] = tokenizer(topic, truncation=True, padding='max_length', max_length=32).input_ids
    # context_batch['topic'] = [[token_id if token_id != tokenizer.pad_token_id else -100 for token_id in topic] for topic
    #               in context_batch['topic']]

    if task == 'know':
        target_knowledge = target_knowledge.tolist()
        candidate_indice = [[know] + negative_sampler(args, know) for know in target_knowledge]
        # candidate_knowledge = tokenizer([args.knowledgeDB[idx] for idx in candidate_indice], truncation=True, padding='max_length', max_length=args.max_length)
        candidate_knowledge_token = [[tokenizer(args.knowledgeDB[i], truncation=True, padding='max_length', max_length=args.max_length).input_ids for i in idx] for idx in candidate_indice]
        candidate_knowledge_mask = [[tokenizer(args.knowledgeDB[i], truncation=True, padding='max_length', max_length=args.max_length).attention_mask for i in idx] for idx in candidate_indice]
        context_batch['candidate_indice'] = candidate_indice  # 이미 Tensor로 받음
        context_batch['candidate_knowledge_token'] = candidate_knowledge_token
        context_batch['candidate_knowledge_mask'] = candidate_knowledge_mask
        # [target, cand1, cand2, cand3, cand4]

    for k, v in context_batch.items():
        if not isinstance(v, torch.Tensor):
            context_batch[k] = torch.as_tensor(v, device=args.device)
            # context_batch[k] = torch.as_tensor(v)
    return context_batch


class KnowledgeDataset(Dataset):
    def __init__(self, args, knowledgeDB, tokenizer):
        super(Dataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.knowledgeDB = knowledgeDB
        self.data_samples = []

    def __getitem__(self, item):
        data = self.knowledgeDB[item]
        tokenized_data = self.tokenizer(data,
                                        max_length=self.max_length,
                                        padding='max_length',
                                        truncation=True,
                                        add_special_tokens=True)
        tokens = torch.LongTensor(tokenized_data.input_ids)
        mask = torch.LongTensor(tokenized_data.attention_mask)
        return tokens, mask

    def __len__(self):
        return len(self.knowledgeDB)


class Retriever(nn.Module):
    def __init__(self, args, query_bert):
        super(Retriever, self).__init__()
        self.args = args
        self.query_bert = query_bert  # Knowledge text 처리를 위한 BERT
        self.hidden_size = query_bert.config.hidden_size
        self.topic_proj = nn.Linear(self.hidden_size, args.topic_num)

    def forward(self, token_seq, mask):
        if self.args.usebart:
            dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask, output_hidden_states=True).decoder_hidden_states[-1][:, 0, :].squeeze(1)
        else:
            dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        dialog_emb = self.proj(dialog_emb)
        return dialog_emb

    def generation(self, token_seq, mask, labels):
        outputs = self.query_bert(input_ids=token_seq, attention_mask=mask, labels=labels, output_hidden_states=True)
        return outputs[0]

    def knowledge_retrieve(self, token_seq, mask, candidate_knowledge_token, candidate_knowledge_mask):
        """
        Args: 뽑아준 negative에 대해서만 dot-product
            token_seq: [B, L]
            mask: [B, L]
            candidate_knowledge_token: [B, K+1, L]
            candidate_knowledge_mask: [B, K+1, L]
        Returns:
        """
        batch_size = mask.size(0)

        # dot-product
        if self.args.usebart:
            dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask, output_hidden_states=True).decoder_hidden_states[-1][:, 0, :].squeeze(1)
        else:
            dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]

        # dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        candidate_knowledge_token = candidate_knowledge_token.view(-1, self.args.max_length)  # [B*(K+1), L]
        candidate_knowledge_mask = candidate_knowledge_mask.view(-1, self.args.max_length)  # [B*(K+1), L]

        knowledge_index = self.query_bert(input_ids=candidate_knowledge_token, attention_mask=candidate_knowledge_mask).last_hidden_state[:, 0, :]
        knowledge_index = knowledge_index.view(batch_size, -1, dialog_emb.size(-1))
        logit = torch.sum(dialog_emb.unsqueeze(1) * knowledge_index, dim=2)  # [B, 1, d] * [B, K+1, d] = [B, K+1]
        return logit


class DialogDatasetKnow(Dataset):  # knowledge용 데이터셋
    def __init__(self, args, conversation_sample, knowledgeDB, tokenizer, task):
        super(Dataset, self).__init__()
        self.args = args
        self.task = task
        self.raw_data = conversation_sample
        self.tokenizer = tokenizer
        self.knowledgeDB = knowledgeDB
        self.augmented_raw_sample = self.augment_raw_dataset(conversation_sample)

    def augment_raw_dataset(self, raw_data):
        train_sample = []
        for ij in range(len(raw_data)):
            conversation = raw_data[ij]
            augmented_dialog = []
            for i in range(len(conversation['dialog'])):
                role = conversation['role_seq'][i]
                if role == 'System' and len(augmented_dialog) > 0 and conversation['knowledge_seq'][i] != '':
                    flatten_dialog = self.tokenizer.sep_token.join(augmented_dialog)
                    train_sample.append({'dialog': flatten_dialog,
                                         'user_profile': conversation['user_profile'],
                                         'response': conversation['dialog'][i],
                                         'type': conversation['type'][i],
                                         'topic': conversation['topic'][i],
                                         'situation': conversation['situation'],
                                         'target_knowledge': self.knowledgeDB.index(conversation['knowledge_seq'][i])})
                augmented_dialog.append(conversation['dialog'][i])
        return train_sample

    def negative_sampler(self, target_knowledge):
        # candidate_entity = self.knowledgeDB[target_knowledge][0]
        # candiate_all_list = self.knowledgeDB_entity_values[candidate_entity]
        # negative_indice = random.choices(candiate_all_list, k=self.args.negative_num if len(candiate_all_list) > self.args.negative_num else len(candiate_all_list))
        total_knowledge_num = self.args.knowledge_num
        negative_indice = []
        while len(negative_indice) < self.args.negative_num:
            negative_idx = random.randint(0, total_knowledge_num - 1)
            if (negative_idx not in negative_indice) and (negative_idx != target_knowledge):
                negative_indice.append(negative_idx)
        return negative_indice

    def __getitem__(self, idx):  # TODO 구현 전
        data = self.augmented_raw_sample[idx]
        cbdicKeys = ['dialog', 'user_profile', 'response', 'type', 'topic', 'situation', 'target_knowledge']
        dialog, user_profile, response, type, topic, situation, target_knowledge_idx = [data[i] for i in cbdicKeys]

        context_batch = defaultdict()
        if self.task == 'know':
            prefix = '<situation>' + situation + '<type>' + type + '<topic>' + topic + "predict the next goal:"

        input_sentence = prefix + '<dialog>' + dialog
        input_sentence = self.tokenizer(input_sentence, add_special_tokens=False).input_ids
        topic_prompt = self.tokenizer.encode('predict the next goal: ')[1:]
        input_sentence = [self.tokenizer.cls_token_id] + input_sentence[-self.args.max_length + len(topic_prompt) + 1:] + topic_prompt
        input_sentence = input_sentence + [self.tokenizer.pad_token_id] * (self.args.max_length - len(input_sentence))
        context_batch['dialog_token'] = torch.LongTensor(input_sentence).to(self.args.device)
        attention_mask = context_batch['dialog_token'].ne(self.tokenizer.pad_token_id)
        context_batch['dialog_mask'] = attention_mask
        context_batch['response'] = self.tokenizer(response,
                                                   add_special_tokens=True,
                                                   max_length=self.args.max_length,
                                                   padding='max_length',
                                                   truncation=True).input_ids

        context_batch['type'] = self.args.goalDic['str'][type]  # index로 바꿈
        context_batch['topic_idx'] = self.args.topicDic['str'][topic]  # index로 바꿈
        context_batch['topic'] = self.tokenizer(topic, truncation=True, padding='max_length', max_length=32).input_ids

        target_knowledge = self.args.knowledgeDB[target_knowledge_idx]

        candidate_indice = [target_knowledge_idx] + negative_sampler(self.args, target_knowledge_idx)
        # candidate_knowledge = tokenizer([args.knowledgeDB[idx] for idx in candidate_indice], truncation=True, padding='max_length', max_length=args.max_length)
        candidate_knowledge_token = self.tokenizer([self.args.knowledgeDB[idx] for idx in candidate_indice], truncation=True, padding='max_length', max_length=self.args.max_length).input_ids
        candidate_knowledge_mask = self.tokenizer([self.args.knowledgeDB[idx] for idx in candidate_indice], truncation=True, padding='max_length', max_length=self.args.max_length).attention_mask
        context_batch['candidate_indice'] = candidate_indice  # 이미 Tensor로 받음
        context_batch['candidate_knowledge_token'] = candidate_knowledge_token
        context_batch['candidate_knowledge_mask'] = candidate_knowledge_mask

        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.args.device)
                # context_batch[k] = torch.as_tensor(v)
        return context_batch

    def __len__(self):
        return len(self.augmented_raw_sample)


def main():
    # TH 작업 main
    args = parseargs()
    # args.data_cache = False
    args.who = "TH"
    args.bert_name = 'facebook/bart-base'
    args.batch_size = 2

    checkPath(args.log_dir)
    checkPath(args.model_dir)

    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(args.log_dir, f'{args.time}_{args.log_name + "_"}log.txt'), filemode='a', format='%(asctime)s: %(levelname)s: %(message)s', datefmt='%Y/%m/%d_%p_%I:%M:%S ')
    logging.info('Commend: {}'.format(', '.join(map(str, sys.argv))))
    # Model cached load
    checkPath(os.path.join("cache", args.bert_name))

    bert_model = BartForConditionalGeneration.from_pretrained(args.bert_name, cache_dir=os.path.join("cache", args.bert_name))

    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
    tokenizer.add_special_tokens(bert_special_tokens_dict)  # [TH] add bert special token (<dialog>, <topic> , <type>)
    bert_model.resize_token_embeddings(len(tokenizer))
    args.hidden_size = bert_model.config.hidden_size  # BERT large 쓸 때 대비

    topicDic = readDic(os.path.join(args.data_dir, "topic2id.txt"))
    goalDic = readDic(os.path.join(args.data_dir, "goal2id.txt"))
    args.topicDic = topicDic
    args.goalDic = goalDic
    args.topic_num = len(topicDic['int'])
    args.goal_num = len(goalDic['int'])

    # Read knowledge DB
    # train_knowledgeDB = data.read_pkl(os.path.join(args.data_dir, 'train_knowledge_DB.pickle'))  # TODO: verbalize (TH)
    knowledgeDB = data.read_pkl(os.path.join(args.data_dir, 'knowledgeDB.txt'))  # TODO: verbalize (TH)
    knowledge_data = KnowledgeDataset(args, knowledgeDB, tokenizer)  # knowledge dataset class
    args.knowledge_num = len(knowledgeDB)
    args.knowledgeDB = knowledgeDB

    train_dataset = dataset_reader(args, tokenizer, knowledgeDB, 'train')
    test_dataset = dataset_reader(args, tokenizer, knowledgeDB, 'test')
    train_datamodel = DialogDatasetKnow(args, train_dataset, knowledgeDB, tokenizer, task='know')
    test_datamodel = DialogDatasetKnow(args, test_dataset, knowledgeDB, tokenizer, task='know')

    train_dataloader = DataLoader(train_datamodel, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_datamodel, batch_size=args.batch_size, shuffle=False)

    retriever = Retriever(args, bert_model)
    retriever = retriever.to(args.device)

    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.AdamW(retriever.parameters(), lr=args.lr)

    # train generate task
    for e in range(5):
        train_epoch_loss = 0
        for batch in tqdm(train_dataloader, desc="Type_Train", bar_format=' {l_bar} | {bar:23} {r_bar}'):
            retriever.train()
            dialog_token = batch['dialog_token']
            dialog_mask = batch['dialog_mask']
            response = batch['response']

            loss = retriever.generation(dialog_token, dialog_mask, response)
            # loss = criterion(dot_score, targets)
            train_epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # test generation task
    all_dialog = []
    all_response = []
    all_generated = []
    for batch in tqdm(test_dataloader, desc="Type_Train", bar_format=' {l_bar} | {bar:23} {r_bar}'):
        retriever.eval()
        dialog_token = batch['dialog_token']
        dialog_mask = batch['dialog_mask']
        response = batch['response']

        batch_size = dialog_token.shape[0]
        generated = retriever.query_bert.generate(input_ids=dialog_token,
                                                  attention_mask=dialog_mask,
                                                  max_length=50)
        decoded_generated = tokenizer.batch_decode(generated, skip_special_tokens=True)
        all_generated.append(decoded_generated)
        all_response.append(tokenizer.batch_decode(response, skip_special_tokens=True))
        all_dialog.append(tokenizer.batch_decode(dialog_token, skip_special_tokens=True))

    with open('response_write.txt', 'w', encoding='UTF-8') as f:
        for (a,b,c) in zip(all_dialog, all_response, all_generated):
            f.write('[DIALOG]\t%s\n[RESPONSE]\t%s\n[GENERATED]\t%s\n', a,b,c)

    # if args.saved_model_path == '':
    #     train_retriever_idx(args, train_dataloader, knowledge_data, retriever)  # [TH] <topic> 추가됐으니까 재학습
    # else:
    #     retriever.load_state_dict(torch.load(os.path.join(args.model_dir, args.saved_model_path)))
    #
    # eval_know(args, test_dataloader, retriever, knowledge_data, knowledgeDB, tokenizer)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리


if __name__ == "__main__":
    main()
    # logging.info('Commend: {}'.format(', '.join(map(str, sys.argv))))
