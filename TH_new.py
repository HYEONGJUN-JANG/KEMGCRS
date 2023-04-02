import copy
import json
import sys
import logging
from collections import defaultdict
import random

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BartForConditionalGeneration, GPT2LMHeadModel, GPT2Config
import data
from config import bert_special_tokens_dict, gpt_special_tokens_dict
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
    if tokenizer.eos_token is not None:
        eos_token = tokenizer.eos_token
    else:
        eos_token=tokenizer.sep_token
    for ij in range(len(raw_data)):
        conversation = raw_data[ij]
        augmented_dialog = []
        for i in range(len(conversation['dialog'])):
            role = conversation['role_seq'][i]
            utterance = conversation['dialog'][i] + eos_token

            if role == 'System' and len(augmented_dialog) > 0 and conversation['knowledge_seq'][i] != '':
                flatten_dialog = ''.join(augmented_dialog)
                train_sample.append({'dialog': flatten_dialog,
                                     'user_profile': conversation['user_profile'],
                                     'response': utterance,
                                     'type': conversation['type'][i],
                                     'topic': conversation['topic'][i],
                                     'situation': conversation['situation'],
                                     'target_knowledge': knowledgeDB.index(conversation['knowledge_seq'][i])})
            augmented_dialog.append(utterance)
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
    def __init__(self, args, query_bert=None, gpt_model=None):
        super(Retriever, self).__init__()
        self.args = args
        self.query_bert = query_bert  # Knowledge text 처리를 위한 BERT
        if args.know_ablation == 'negative_sampling':
            self.key_bert = query_bert
        else:
            self.key_bert = copy.deepcopy((query_bert))
            self.key_bert.requires_grad = False

        self.gpt_model = gpt_model
        self.hidden_size = args.hidden_size
        self.topic_proj = nn.Linear(self.hidden_size, args.topic_num)
        self.know_proj = nn.Linear(self.hidden_size, self.args.knowledge_num)

    def forward(self, token_seq, mask):
        if self.args.usebart:
            dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask, output_hidden_states=True).decoder_hidden_states[-1][:, 0, :].squeeze(1)
        else:
            dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        dialog_emb = self.proj(dialog_emb)
        return dialog_emb

    def generation(self, token_seq, mask, labels):
        outputs = self.gpt_model(input_ids=token_seq, attention_mask=mask, labels=labels)
        # outputs = self.gpt_model(input_ids=token_seq, labels=labels)

        return outputs[0]

    def compute_know_score(self, token_seq, mask, knowledge_index):
        """
        eval_know.computing_score에서
        모든 key vector에서 올라온 벡터를 통해 계산처리
        """
        if self.args.usebart:
            dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask, output_hidden_states=True).decoder_hidden_states[-1][:, 0, :].squeeze(1)
        else:
            dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
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

        if self.args.usebart:
            dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask, output_hidden_states=True).decoder_hidden_states[-1][:, 0, :].squeeze(1)
        else:
            dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]

        # dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]


        if self.args.know_ablation == 'mlp':
            logit = self.know_proj(dialog_emb)
        elif self.args.know_ablation == 'negative_sampling':
            candidate_knowledge_token = candidate_knowledge_token.view(-1, self.args.max_length)  # [B*(K+1), L]
            candidate_knowledge_mask = candidate_knowledge_mask.view(-1, self.args.max_length)  # [B*(K+1), L]

            if self.args.usebart:
                knowledge_index = self.query_bert(input_ids=candidate_knowledge_token, attention_mask=candidate_knowledge_mask, output_hidden_states=True).decoder_hidden_states[-1][:, 0, :].squeeze(1)
            else:
                knowledge_index = self.query_bert(input_ids=candidate_knowledge_token, attention_mask=candidate_knowledge_mask).last_hidden_state[:, 0, :]  # [B, d]

            knowledge_index = knowledge_index.view(batch_size, -1, dialog_emb.size(-1))
            logit = torch.sum(dialog_emb.unsqueeze(1) * knowledge_index, dim=2)  # [B, 1, d] * [B, K+1, d] = [B, K+1]

        return logit


class DialogDataset(Dataset):  # knowledge용 데이터셋
    def __init__(self, args, data_sample, knowledgeDB, tokenizer, task, mode='train'):
        super(Dataset, self).__init__()
        self.args = args
        self.task = task
        self.tokenizer = tokenizer
        self.knowledgeDB = knowledgeDB
        self.augmented_raw_sample = data_sample
        self.mode = mode

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
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        context_batch = defaultdict()
        if self.task == 'know':
            prefix = '<type>' + type + '<topic>' + topic
            topic_prompt = self.tokenizer.encode('predict the next knowledge: ')[1:]
        elif self.task == 'resp':
            # prefix = '<knowledge>' + self.args.knowledgeDB[target_knowledge_idx]
            prefix = ''
            topic_prompt = self.tokenizer.encode('predict the next response: ')[1:]

        prefix_encoding = self.tokenizer.encode(prefix)[1:][:30]

        input_sentence = self.tokenizer('<dialog>' + dialog, add_special_tokens=False).input_ids
        if self.tokenizer.cls_token_id is not None:
            input_sentence = [self.tokenizer.cls_token_id] + prefix_encoding + input_sentence[-(self.args.max_length - len(topic_prompt) - len(prefix_encoding) - 1):] + topic_prompt
        else:
            input_sentence = prefix_encoding + input_sentence[-(self.args.max_length - len(topic_prompt) - len(prefix_encoding)):] + topic_prompt

        if self.mode != 'generate':
            input_sentence = input_sentence + [pad_token_id] * (self.args.max_length - len(input_sentence))
        else:
            input_sentence = [pad_token_id] * (self.args.max_length - len(input_sentence)) + input_sentence

        context_batch['input_ids'] = torch.LongTensor(input_sentence).to(self.args.device)
        attention_mask = context_batch['input_ids'].ne(pad_token_id)
        context_batch['attention_mask'] = attention_mask
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


class GenerationDataset(Dataset):  # knowledge용 데이터셋
    def __init__(self, args, data_sample, knowledgeDB, tokenizer, mode='train', knowledge=False):
        super(Dataset, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.knowledgeDB = knowledgeDB
        self.augmented_raw_sample = data_sample
        self.mode = mode
        self.knowledge = knowledge
        print(knowledge)
        self.generate_prompt_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('System:'))

    def __getitem__(self, idx):  # TODO 구현 전
        data = self.augmented_raw_sample[idx]
        cbdicKeys = ['dialog', 'user_profile', 'response', 'type', 'topic', 'situation', 'target_knowledge']
        dialog, user_profile, response, type, topic, situation, target_knowledge_idx = [data[i] for i in cbdicKeys]
        pad_token_id = self.tokenizer.pad_token_id

        context_batch = defaultdict()
        resp_batch = []
        context_len_batch = []

        prefix = ''
        topic_prompt = self.tokenizer.encode('predict the next response: ')[1:]

        prefix_encoding = self.tokenizer.encode(prefix)[1:][:30]
        knowledge_text = self.knowledgeDB[target_knowledge_idx]

        max_knowledge_length=30
        if self.knowledge:
            knowledge_text = self.tokenizer('<knowledge>'+self.knowledgeDB[target_knowledge_idx], max_length=max_knowledge_length,
                                truncation=True).input_ids
        else:
            knowledge_text = []

        dialog = self.tokenizer('<dialog>'+dialog, max_length=self.args.max_length-len(knowledge_text),
                                truncation=True).input_ids
        dialog = knowledge_text+dialog

        if self.mode == 'train':
            response = self.tokenizer(response, max_length=self.args.max_gen_length,
                                      truncation=True).input_ids

            # self.tokenizer.padding_side = 'right'
            max_length = self.args.max_length + self.args.max_gen_length
            context_ids = dialog + response
            context_ids = context_ids[-max_length:]
            context_ids = context_ids + [pad_token_id] * (max_length - len(context_ids))
            # resp_batch = [token_id if token_id != self.tokenizer.pad_token_id else -100 for token_id in context_ids]
            resp_batch = context_ids

            context_batch['input_ids'] = torch.LongTensor(context_ids)
            context_batch['attention_mask'] = torch.ne(context_batch['input_ids'], pad_token_id)
            context_batch['response'] = torch.LongTensor(resp_batch)

        elif self.mode == 'test':
            # self.tokenizer.padding_side = 'left'

            context_ids = dialog[-(self.args.max_length - len(self.generate_prompt_ids)):]
            context_len_batch = len([token for token in context_ids if token != pad_token_id])
            context_ids += self.generate_prompt_ids

            context_ids = [pad_token_id] * (self.args.max_length - len(context_ids)) + context_ids
            context_batch['input_ids'] = torch.LongTensor(context_ids)
            context_batch['attention_mask'] = torch.ne(context_batch['input_ids'], pad_token_id)

            context_batch['response'] = response
            context_batch['context_len'] = context_len_batch

        # for k, v in context_batch.items():
        #     if not isinstance(v, torch.Tensor):
        #         context_batch[k] = torch.as_tensor(v, device=self.args.device)
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

        if args.usebart:
            knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).decoder_hidden_states[-1][:, 0, :].squeeze(1)
        else:
            knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]
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
    for batch in tqdm(test_dataloader, desc="Knowledge_Test", bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):  # TODO: Knowledge task 분리중
        dialog_token = batch['input_ids']
        dialog_mask = batch['attention_mask']
        response = batch['response']
        candidate_knowledge_token = batch['candidate_knowledge_token']  # [B,5,256]
        type_idx = batch['type']
        topic_idx = batch['topic_idx']
        candidate_knowledge_mask = batch['candidate_knowledge_mask']  # [B,5,256]
        target_knowledge = candidate_knowledge_token[:, 0, :]
        target_knowledge_idx = int(batch['candidate_indice'][:, 0])

        # tokenizer.batch_decode(dialog_token, skip_special_tokens=True)  # 'dialog context'
        # print([knowledgeDB[idx] for idx in target_knowledge]) # target knowledge
        # dot_score = retriever.compute_score(response_token, response_mask, knowledge_index)
        if args.know_ablation == 'negative_sampling':
            dot_score = retriever.compute_know_score(dialog_token, dialog_mask, knowledge_index)
        elif args.know_ablation == 'mlp':
            dot_score = retriever.knowledge_retrieve(dialog_token, dialog_mask, candidate_knowledge_token, candidate_knowledge_mask)

        top_candidate = torch.topk(dot_score, k=args.know_topk, dim=1).indices  # [B, K]

        input_text = '||'.join(tokenizer.batch_decode(dialog_token, skip_special_tokens=True))
        target_knowledge_text = tokenizer.batch_decode(target_knowledge, skip_special_tokens=True)  # target knowledge
        retrieved_knowledge_text = [knowledgeDB[idx].lower() for idx in top_candidate[0]]  # list
        correct = target_knowledge_idx in top_candidate

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
    # args.bert_name = 'facebook/bart-base'
    args.task = 'know'
    # args.usebart = True
    args.max_gen_length = 50
    args.know_ablation = 'mlp'

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
    args.knowledge_num = len(knowledgeDB)
    args.knowledgeDB = knowledgeDB

    train_dataset_raw = dataset_reader(args, 'train')
    test_dataset_raw = dataset_reader(args, 'test')

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
        args.bert_name = 'bert-base-uncased'
        args.usebart = False

        bert_model = AutoModel.from_pretrained(args.bert_name, cache_dir=os.path.join("cache", args.bert_name))
        tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
        tokenizer.add_special_tokens(bert_special_tokens_dict)  # [TH] add bert special token (<dialog>, <topic> , <type>)
        bert_model.resize_token_embeddings(len(tokenizer))
        args.hidden_size = bert_model.config.hidden_size  # BERT large 쓸 때 대비

        retriever = Retriever(args, bert_model)
        retriever = retriever.to(args.device)
        optimizer = optim.AdamW(retriever.parameters(), lr=args.lr)

        knowledge_data = KnowledgeDataset(args, knowledgeDB, tokenizer)  # knowledge dataset class
        args.knowledge_num = len(knowledgeDB)
        args.knowledgeDB = knowledgeDB

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
                dialog_token = batch['input_ids']
                dialog_mask = batch['attention_mask']
                # response = batch['response']
                candidate_knowledge_token = batch['candidate_knowledge_token']  # [B,5,256]
                candidate_knowledge_mask = batch['candidate_knowledge_mask']  # [B,5,256]
                target_knowledge = candidate_knowledge_token[:, 0, :]
                target_knowledge_idx = torch.stack([idx[0] for idx in batch['candidate_indice']])

                logit = retriever.knowledge_retrieve(dialog_token, dialog_mask, candidate_knowledge_token, candidate_knowledge_mask)
                # loss = (-torch.log_softmax(logit, dim=1).select(dim=1, index=0)).mean()
                loss = criterion(dot_score, target_knowledge_idx)
                train_epoch_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch: {epoch}\nTrain Loss: {train_epoch_loss}")

        eval_know(args, test_dataloader, retriever, knowledge_data, knowledgeDB, tokenizer)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리


if __name__ == "__main__":
    main()
    # logging.info('Commend: {}'.format(', '.join(map(str, sys.argv))))
