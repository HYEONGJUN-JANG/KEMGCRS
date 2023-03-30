import json
import sys
import logging
from collections import defaultdict
import random

import torch
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


def process_augment_sample(raw_data, tokenizer, knowledgeDB):
    train_sample = []
    for ij in range(len(raw_data)):
        conversation = raw_data[ij]
        augmented_dialog = []
        for i in range(len(conversation['dialog'])):
            role = conversation['role_seq'][i]
            if role == 'System' and len(augmented_dialog) > 0 and conversation['knowledge_seq'][i] != '':
                flatten_dialog = tokenizer.sep_token.join(augmented_dialog)
                train_sample.append({'dialog': flatten_dialog,
                                     'user_profile': conversation['user_profile'],
                                     'response': conversation['dialog'][i],
                                     'type': conversation['type'][i],
                                     'topic': conversation['topic'][i],
                                     'situation': conversation['situation'],
                                     'target_knowledge': knowledgeDB.index(conversation['knowledge_seq'][i])})
            augmented_dialog.append(conversation['dialog'][i])
    return train_sample


def dataset_reader(args, data_name='train'):
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

    def compute_know_score(self, token_seq, mask, knowledge_index):
        """
        eval_know.computing_score에서
        모든 key vector에서 올라온 벡터를 통해 계산처리
        """
        if self.args.usebart: dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask, output_hidden_states=True).decoder_hidden_states[-1][:,0,:].squeeze(1)
        else: dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        # dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        dot_score = torch.matmul(dialog_emb, knowledge_index.transpose(1, 0))  # [B, N]
        return dot_score

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
        # if self.args.usebart:
        #     dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask, output_hidden_states=True).decoder_hidden_states[-1][:, 0, :].squeeze(1)
        # else:
        #     dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]

        if self.args.usebart: dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask, output_hidden_states=True).decoder_hidden_states[-1][:,0,:].squeeze(1)
        else: dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]

        # dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        candidate_knowledge_token = candidate_knowledge_token.view(-1, self.args.max_length)  # [B*(K+1), L]
        candidate_knowledge_mask = candidate_knowledge_mask.view(-1, self.args.max_length)  # [B*(K+1), L]

        if self.args.usebart:
            knowledge_index = self.query_bert(input_ids=candidate_knowledge_token, attention_mask=candidate_knowledge_mask, output_hidden_states=True).decoder_hidden_states[-1][:, 0, :].squeeze(1)
        else:
            knowledge_index = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]

        knowledge_index = knowledge_index.view(batch_size, -1, dialog_emb.size(-1))
        logit = torch.sum(dialog_emb.unsqueeze(1) * knowledge_index, dim=2)  # [B, 1, d] * [B, K+1, d] = [B, K+1]
        return logit


class DialogDataset(Dataset):  # knowledge용 데이터셋
    def __init__(self, args, data_sample, knowledgeDB, tokenizer, task):
        super(Dataset, self).__init__()
        self.args = args
        self.task = task
        self.tokenizer = tokenizer
        self.knowledgeDB = knowledgeDB
        self.augmented_raw_sample = data_sample

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
            prefix = '<type>' + type + '<topic>' + topic
            topic_prompt = self.tokenizer.encode('predict the next knowledge: ')[1:]
        elif self.task == 'resp':
            prefix = '<knowledge>' + self.args.knowledgeDB[target_knowledge_idx]
            topic_prompt = self.tokenizer.encode('predict the next response: ')[1:]

        prefix_encoding = self.tokenizer.encode(prefix)[1:][:30]

        input_sentence = self.tokenizer('<dialog>'+dialog, add_special_tokens=False).input_ids
        input_sentence = [self.tokenizer.cls_token_id] + prefix_encoding + input_sentence[-(self.args.max_length - len(topic_prompt) - len(prefix_encoding) - 1):] + topic_prompt
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

        # target_knowledge = self.args.knowledgeDB[target_knowledge_idx]

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


def knowledge_reindexing(args, knowledge_data, retriever):
    # 모든 know_index를 버트에 태움
    print('...knowledge indexing...')
    knowledgeDataLoader = DataLoader(
        knowledge_data,
        batch_size=args.batch_size
    )
    knowledge_index = []

    for batch in tqdm(knowledgeDataLoader):
        input_ids = batch[0].to(args.device)
        attention_mask = batch[1].to(args.device)
        knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).decoder_hidden_states[-1][:, 0, :].squeeze(1)
        knowledge_index.extend(knowledge_emb.cpu().detach())
    knowledge_index = torch.stack(knowledge_index, 0)
    return knowledge_index


def eval_know(args, test_dataloader, retriever, knowledge_data, knowledgeDB, tokenizer):

    # Read knowledge DB
    # knowledge_index = knowledge_reindexing(args, knowledge_data, retriever)
    # knowledge_index = knowledge_index.to(args.device)
    jsonlineSave = []
    # bert_model = bert_model.to(args.device)
    knowledge_index = knowledge_reindexing(args, knowledge_data, retriever)
    knowledge_index = knowledge_index.to(args.device)

    cnt = 0
    for batch in tqdm(test_dataloader, desc="Knowledge_Test", bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'): # TODO: Knowledge task 분리중
        dialog_token = batch['dialog_token']
        dialog_mask = batch['dialog_mask']
        response = batch['response']
        candidate_knowledge_token = batch['candidate_knowledge_token']  # [B,5,256]
        type_idx = batch['type']
        topic_idx = batch['topic_idx']
        candidate_knowledge_mask = batch['candidate_knowledge_mask']  # [B,5,256]
        target_knowledge = candidate_knowledge_token[:, 0, :]

        # tokenizer.batch_decode(dialog_token, skip_special_tokens=True)  # 'dialog context'
        # print([knowledgeDB[idx] for idx in target_knowledge]) # target knowledge
        # dot_score = retriever.compute_score(response_token, response_mask, knowledge_index)
        dot_score = retriever.compute_know_score(dialog_token, dialog_mask, knowledge_index)
        top_candidate = torch.topk(dot_score, k=args.know_topk, dim=1).indices  # [B, K]

        input_text = '||'.join(tokenizer.batch_decode(dialog_token, skip_special_tokens=True))
        target_knowledge_text = tokenizer.batch_decode(target_knowledge, skip_special_tokens=True)  # target knowledge
        retrieved_knowledge_text = [knowledgeDB[idx] for idx in top_candidate[0]]  # list
        correct = target_knowledge_text in retrieved_knowledge_text
        response = '||'.join(tokenizer.batch_decode(response, skip_special_tokens=True))

        type_idx = [args.goalDic['int'][int(idx)] for idx in type_idx]
        topic_idx = [args.topicDic['int'][int(idx)] for idx in topic_idx]

        jsonlineSave.append({'goal_type': type_idx[0], 'topic': topic_idx[0], 'tf': correct, 'dialog': input_text, 'target': '||'.join(target_knowledge_text), 'response': response, "predict5": retrieved_knowledge_text})
        cnt += 1

    # TODO HJ: 입출력 저장 args처리 필요시 args.save_know_output 에 store_true 옵션으로 만들 필요
    write_pkl(obj=jsonlineSave, filename='jsonline.pkl')  # 입출력 저장
    save_json(args, f"{args.time}_{args.log_name}_inout", jsonlineSave)
    print('done')


def main():
    # TH 작업 main
    args = parseargs()
    # args.data_cache = False
    args.who = "TH"
    args.bert_name = 'bert-base-uncased'

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

    # train_dataset_raw = dataset_reader(args, 'train')
    # test_dataset_raw = dataset_reader(args, 'test')
    # train_dataset = process_augment_sample(train_dataset_raw, tokenizer, knowledgeDB)
    # test_dataset = process_augment_sample(test_dataset_raw, tokenizer, knowledgeDB)
    # train_datamodel_resp = DialogDataset(args, train_dataset, knowledgeDB, tokenizer, task='resp')
    # test_datamodel_resp = DialogDataset(args, test_dataset, knowledgeDB, tokenizer, task='resp')
    #
    # train_dataloader = DataLoader(train_datamodel_resp, batch_size=args.batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_datamodel_resp, batch_size=args.batch_size, shuffle=False)
    #
    # generator = Retriever(args, bert_model)
    # generator = generator.to(args.device)
    # criterion = nn.CrossEntropyLoss().to(args.device)
    # optimizer = optim.AdamW(generator.parameters(), lr=args.lr)
    # train generate task
    # if args.saved_model_path == '':
    #     for epoch in range(args.num_epochs):
    #         train_epoch_loss = 0
    #         for batch in tqdm(train_dataloader, desc="Generate_Train", bar_format=' {l_bar} | {bar:23} {r_bar}'):
    #             generator.train()
    #             dialog_token = batch['dialog_token']
    #             dialog_mask = batch['dialog_mask']
    #             response = batch['response']
    #
    #             loss = generator.generation(dialog_token, dialog_mask, response)
    #             # loss = criterion(dot_score, targets)
    #             train_epoch_loss += loss
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #         print(f"Epoch: {epoch}\nTrain Loss: {train_epoch_loss}")
    #     torch.save(generator.state_dict(), os.path.join(args.model_dir, f"{args.time}_{args.model_name}_bin.pt"))  # TIME_MODELNAME 형식
    #
    #     # test generation task
    #     all_dialog = []
    #     all_response = []
    #     all_generated = []
    #     for batch in tqdm(test_dataloader, desc="Generate Test", bar_format=' {l_bar} | {bar:23} {r_bar}'):
    #         generator.eval()
    #         dialog_token = batch['dialog_token']
    #         dialog_mask = batch['dialog_mask']
    #         response = batch['response']
    #
    #         batch_size = dialog_token.shape[0]
    #         generated = generator.query_bert.generate(input_ids=dialog_token,
    #                                                   attention_mask=dialog_mask,
    #                                                   max_length=50)
    #         decoded_generated = tokenizer.batch_decode(generated, skip_special_tokens=True)
    #         all_generated.extend(decoded_generated)
    #         all_response.extend(tokenizer.batch_decode(response, skip_special_tokens=True))
    #         all_dialog.extend(tokenizer.batch_decode(dialog_token, skip_special_tokens=True))
    #
    #     with open(f"response_write_{args.time}_{args.model_name}.txt", 'w', encoding='UTF-8') as f:
    #         for (a, b, c) in zip(all_dialog, all_response, all_generated):
    #             f.write('[DIALOG]\t%s\n[RESPONSE]\t%s\n[GENERATED]\t%s\n' % (a, b, c))
    #             f.write('-------------------------------------------\n')
    # else:
    #     generator.load_state_dict(torch.load(os.path.join(args.model_dir, args.saved_model_path)))
    args.bert_name = 'args.bert_name'
    args.usebart = False

    bert_model = AutoModel.from_pretrained(args.bert_name, cache_dir=os.path.join("cache", args.bert_name))
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
    tokenizer.add_special_tokens(bert_special_tokens_dict)  # [TH] add bert special token (<dialog>, <topic> , <type>)
    bert_model.resize_token_embeddings(len(tokenizer))
    args.hidden_size = bert_model.config.hidden_size  # BERT large 쓸 때 대비

    retriever = Retriever(args, bert_model)
    retriever = retriever.to(args.device)
    optimizer = optim.AdamW(retriever.parameters(), lr=args.lr)

    train_dataset_raw = dataset_reader(args, 'train')
    test_dataset_raw = dataset_reader(args, 'test')
    train_dataset = process_augment_sample(train_dataset_raw, tokenizer, knowledgeDB)
    test_dataset = process_augment_sample(test_dataset_raw, tokenizer, knowledgeDB)

    train_datamodel_know = DialogDataset(args, train_dataset, knowledgeDB, tokenizer, task='know')
    test_datamodel_know = DialogDataset(args, test_dataset, knowledgeDB, tokenizer, task='know')
    train_dataloader = DataLoader(train_datamodel_know, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_datamodel_know, batch_size=1, shuffle=False)

    for epoch in range(args.num_epochs):
        train_epoch_loss = 0
        for batch in tqdm(train_dataloader, desc="Knowledge_Train", bar_format=' {l_bar} | {bar:23} {r_bar}'):
            retriever.train()
            dialog_token = batch['dialog_token']
            dialog_mask = batch['dialog_mask']
            # response = batch['response']
            candidate_knowledge_token = batch['candidate_knowledge_token']  # [B,5,256]
            candidate_knowledge_mask = batch['candidate_knowledge_mask']  # [B,5,256]

            logit = retriever.knowledge_retrieve(dialog_token, dialog_mask, candidate_knowledge_token, candidate_knowledge_mask)
            loss = (-torch.log_softmax(logit, dim=1).select(dim=1, index=0)).mean()
            # loss = criterion(dot_score, targets)
            train_epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}\nTrain Loss: {train_epoch_loss}")

    eval_know(args, test_dataloader, retriever, knowledge_data, knowledgeDB, tokenizer)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리

    # if args.saved_model_path == '':
    #     train_retriever_idx(args, train_dataloader, knowledge_data, retriever)  # [TH] <topic> 추가됐으니까 재학습
    # else:
    #     retriever.load_state_dict(torch.load(os.path.join(args.model_dir, args.saved_model_path)))
    #
    # eval_know(args, test_dataloader, retriever, knowledge_data, knowledgeDB, tokenizer)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리


if __name__ == "__main__":
    main()
    # logging.info('Commend: {}'.format(', '.join(map(str, sys.argv))))
