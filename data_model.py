import random
from collections import defaultdict

import torch
from responses import target
from torch.utils.data import Dataset
import numpy as np


# def user_profile_setting(ufDic: dict) -> str:  # Accepted 제외하고 다 버림
#     uf = ''
#     for i, key in enumerate(ufDic.keys()):
#         one = ufDic[key]
#         if i == 0 or key[0].lower() != "a":
#             pass
#         else:
#             uf += ' | '
#         if type(one) == list:
#             uf += f"{key}: {', '.join(one[:-5])}"
#         else:
#             uf += f"{key}: {one}"
#     return uf


def user_profile_setting(ufDic: dict) -> str:  # Accepted 제외하고 다 버림
    uf = '<user_profile> '
    for i, key in enumerate(ufDic.keys()):
        one = ufDic[key]
        if i == 0 or key[0].lower() != "a": continue
        if type(one) == list:
            uf += f"{key}: {', '.join(one[-5:])} |"
        else:
            uf += f"{key}: {one}|"
    return uf


def truncationPadding(input_ids, max_length, prefix=[], suffix=[]):
    truncate_size = max_length - len(prefix) - len(suffix)
    # input_ids = prefix + input_ids[-truncate_size:] + suffix
    # input_ids = input_ids + [0] * (max_length - len(input_ids))
    # return input_ids
    if truncate_size <= len(input_ids):
        input_ids = prefix + input_ids[len(input_ids) - truncate_size:] + suffix
    else:
        input_ids = prefix + input_ids + suffix
    return input_ids + [0] * (max_length - len(input_ids))


class GenerationDataset(Dataset):  # knowledge용 데이터셋
    def __init__(self, args, data_sample, knowledgeDB, tokenizer, mode='train', subtask='response'):
        super(Dataset, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.knowledgeDB = knowledgeDB
        self.augmented_raw_sample = data_sample
        self.mode = mode
        self.subtask = subtask
        self.generate_prompt_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('System:'))

    def negative_sampler(self, target_knowledge, candidate_knowledges):
        negative_indice = []
        if len(candidate_knowledges) < self.args.negative_num:
            for idx in range(self.args.negative_num - len(candidate_knowledges)):
                negative_indice.append(0)

        while len(negative_indice) < self.args.negative_num:
            # negative_idx = random.randint(0, total_knowledge_num - 1)
            negative_idx = random.choice(candidate_knowledges)
            if (negative_idx not in negative_indice) and (negative_idx not in target_knowledge):
                negative_indice.append(negative_idx)
        return negative_indice

    def __getitem__(self, idx):  # TODO 구현 전
        data = self.augmented_raw_sample[idx]
        cbdicKeys = ['dialog', 'user_profile', 'response', 'type', 'topic', 'situation', 'target_knowledge', 'candidate_knowledges', 'candidate_confidences']
        dialog, user_profile, response, type, topic, situation, target_knowledge_idx, candidate_knowledges, candidate_confidences = [data[i] for i in cbdicKeys]
        pad_token_id = self.tokenizer.pad_token_id

        context_batch = defaultdict()
        context_batch['goal_type'] = self.tokenizer(type, max_length=self.args.max_gen_length, truncation=True, padding='max_length').input_ids
        resp_batch = []
        context_len_batch = []

        prefix = ''
        if self.subtask == 'know':
            if self.mode == 'train':
                pseudo_negative = self.negative_sampler([target_knowledge_idx], candidate_knowledges)
                pseudo_knowledges = [target_knowledge_idx] + pseudo_negative
                # random.shuffle(pseudo_knowledges)
            else:
                pseudo_knowledges = candidate_knowledges[:5]
            candidate_knowledge_text = [self.knowledgeDB[idx] for idx in pseudo_knowledges]

            related_knowledges = '|'.join(candidate_knowledge_text)
            prompt = self.tokenizer.encode('<knowledge>%s. predict the next %s: ' % (related_knowledges, self.tokenizer.eos_token, self.subtask))[:256]
        else:
            prompt = self.tokenizer.encode('predict the next %s: ' % self.subtask)
        # prefix_encoding = self.tokenizer.encode(prefix)[1:][:30]
        # knowledge_text = self.knowledgeDB[target_knowledge_idx]

        max_knowledge_length = 30
        # if self.knowledge:
        #     knowledge_text = self.tokenizer('<knowledge>' + self.knowledgeDB[target_knowledge_idx], max_length=max_knowledge_length,
        #                                     truncation=True).input_ids
        # else:
        #     knowledge_text = []

        # dialog = self.tokenizer('<dialog>' + dialog, max_length=self.args.max_length - len(prompt), truncation=True).input_ids
        dialog = self.tokenizer('<dialog>' + dialog).input_ids[-(self.args.max_length - len(prompt)):]
        dialog = dialog + prompt

        if self.subtask == 'type':
            label = self.tokenizer(type, max_length=self.args.max_gen_length, truncation=True).input_ids
        elif self.subtask == 'topic':
            label = self.tokenizer(topic, max_length=self.args.max_gen_length, truncation=True).input_ids
        elif self.subtask == 'response':
            label = self.tokenizer(response, max_length=self.args.max_gen_length, truncation=True).input_ids
        elif self.subtask == 'know':
            label = self.tokenizer(self.knowledgeDB[target_knowledge_idx], max_length=self.args.max_gen_length, truncation=True).input_ids

        if self.mode == 'train':
            # self.tokenizer.padding_side = 'right'
            max_length = self.args.max_length + self.args.max_gen_length
            context_ids = dialog + label
            context_ids = context_ids[-max_length:]
            context_ids = context_ids + [pad_token_id] * (max_length - len(context_ids))
            # resp_batch = [token_id if token_id != self.tokenizer.pad_token_id else -100 for token_id in context_ids]
            resp_batch = context_ids

            context_batch['input_ids'] = torch.LongTensor(context_ids)
            context_batch['attention_mask'] = torch.ne(context_batch['input_ids'], pad_token_id)
            context_batch['response'] = torch.LongTensor(resp_batch)

        elif self.mode == 'test':
            # self.tokenizer.padding_side = 'left'

            context_ids = dialog
            # context_ids = dialog[-(self.args.max_length - len(self.generate_prompt_ids)):]
            context_len_batch = len([token for token in context_ids if token != pad_token_id])
            # context_ids += self.generate_prompt_ids

            context_ids = [pad_token_id] * (self.args.max_length - len(context_ids)) + context_ids
            context_batch['input_ids'] = torch.LongTensor(context_ids)
            context_batch['attention_mask'] = torch.ne(context_batch['input_ids'], pad_token_id)

            context_batch['response'] = label + [pad_token_id] * (self.args.max_gen_length - len(label))
            context_batch['context_len'] = context_len_batch

        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.args.device)
        context_batch[k] = torch.as_tensor(v)
        return context_batch

    def __len__(self):
        return len(self.augmented_raw_sample)


def convert_idx_to_docid(idx):
    return f"{idx}"


class DialogDataset(Dataset):  # knowledge용 데이터셋
    def __init__(self, args, data_sample, knowledgeDB, train_knowledgeDB, tokenizer, task, mode='train'):
        super(Dataset, self).__init__()
        self.args = args
        self.task = task
        self.tokenizer = tokenizer
        self.knowledgeDB = knowledgeDB
        self.train_knowledgeDB = train_knowledgeDB
        self.augmented_raw_sample = data_sample
        self.mode = mode

    def negative_sampler(self, target_knowledge, candidate_knowledges):
        # candidate_entity = self.knowledgeDB[target_knowledge][0]
        # candiate_all_list = self.knowledgeDB_entity_values[candidate_entity]
        # negative_indice = random.choices(candiate_all_list, k=self.args.negative_num if len(candiate_all_list) > self.args.negative_num else len(candiate_all_list))
        # total_knowledge_num = self.args.knowledge_num

        negative_indice = []
        if len(candidate_knowledges) < self.args.negative_num:
            for idx in range(self.args.negative_num - len(candidate_knowledges)):
                negative_indice.append(0)

        while len(negative_indice) < self.args.negative_num:
            # negative_idx = random.randint(0, total_knowledge_num - 1)
            negative_idx = random.choice(candidate_knowledges)
            if (negative_idx not in negative_indice) and (negative_idx not in target_knowledge):
                negative_indice.append(negative_idx)
        return negative_indice

    def all_negative(self, candidate_knowledges):
        all_negative = [i for i in range(len(self.knowledgeDB))]
        for candidate in candidate_knowledges:
            all_negative.remove(candidate)
        return all_negative

    def __getitem__(self, idx):  # TODO 구현 전
        data = self.augmented_raw_sample[idx]
        cbdicKeys = ['dialog', 'user_profile', 'response', 'type', 'topic', 'situation', 'target_knowledge', 'candidate_knowledges', 'candidate_confidences']
        dialog, user_profile, response, type, topic, situation, target_knowledge_idx, candidate_knowledges, candidate_confidences = [data[i] for i in cbdicKeys]
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        context_batch = defaultdict()
        if self.args.input_prompt == 'dialog':
            prefix = ''
        elif self.args.input_prompt == 'dialog_goal':
            prefix = '<type>' + type + self.tokenizer.sep_token
        elif self.args.input_prompt == 'dialog_topic':
            prefix = '<topic>' + topic + self.tokenizer.sep_token
        elif self.args.input_prompt == 'dialog_goal_topic':
            prefix = '<type>' + type + '<topic>' + topic + self.tokenizer.sep_token
        elif self.args.input_prompt == 'dialog_topic_profile':
            prefix = '<profile>' + user_profile + '<topic>' + topic + self.tokenizer.sep_token
        elif self.args.input_prompt == 'dialog_type_profile':
            prefix = '<profile>' + user_profile + '<type>' + type + self.tokenizer.sep_token
        else:
            assert Exception

        prefix_encoding = self.tokenizer.encode(prefix)[1:-1][:30]
        input_sentence = self.tokenizer('<dialog>' + dialog, add_special_tokens=False).input_ids

        input_sentence = [self.tokenizer.cls_token_id] + prefix_encoding + input_sentence[-(self.args.max_length - len(prefix_encoding) - 1):]
        input_sentence = input_sentence + [pad_token_id] * (self.args.max_length - len(input_sentence))

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

        # List-wise
        # if target_knowledge_idx in candidate_knowledges:
        #     index = candidate_knowledges.index(target_knowledge_idx)
        #     candidate_knowledges = candidate_knowledges[:index] + candidate_knowledges[index+1:]
        #     candidate_confidences = candidate_confidences[:index] + candidate_confidences[index + 1:]
        # candidate_knowledges = [target_knowledge_idx] + candidate_knowledges
        # candidate_confidences = [100] + candidate_confidences

        # if target_knowledge_idx in candidate_knowledges:
        #     idxinlist = candidate_knowledges.index(target_knowledge_idx)
        #     candidate_knowledges.pop(idxinlist)
        #     candidate_knowledges.insert(0, target_knowledge_idx)
        #     target_confidence = candidate_confidences.pop(idxinlist)
        #     candidate_confidences.insert(0, target_confidence)

        candidate_knowledges_pos = candidate_knowledges[:self.args.pseudo_pos_num]
        candidate_confidences_pos = candidate_confidences[:self.args.pseudo_pos_num]

        # random_idx = random.randrange(min(self.args.pseudo_pos_num, len(candidate_knowledges)))

        ### Positive sampling (Top-K)
        # candidate_knowledges = candidate_knowledges[:self.args.pseudo_pos_num]
        # candidate_confidences = candidate_confidences[:self.args.pseudo_pos_num]
        # random_idx = random.randrange(self.args.pseudo_pos_num)
        # candidate_knowledges = [candidate_knowledges[random_idx]]
        # candidate_confidences = [candidate_confidences[random_idx]]

        ### Positive sampling (pseudo)
        # pseudo_candidate_knowledges = [know for know, conf in zip(candidate_knowledges, candidate_confidences) if conf > 0.1]
        # if len(pseudo_candidate_knowledges) == 0:
        #     candidate_knowledges = [candidate_knowledges[0]]
        #     candidate_confidences = [candidate_confidences[0]]
        # else:
        #     random_idx = random.randrange(len(pseudo_candidate_knowledges))
        #     candidate_knowledges = [pseudo_candidate_knowledges[random_idx]]
        #     candidate_confidences = [pseudo_candidate_knowledges[random_idx]]

        # # Only Top-K
        # candidate_knowledges = [candidate_knowledges[self.args.pseudo_pos_num]]
        # candidate_confidences = [candidate_confidences[self.args.pseudo_pos_num]]

        # candidate_knowledges = candidate_knowledges + [0] * (self.args.pseudo_pos_num - len(candidate_knowledges))
        # candidate_confidences = candidate_confidences + [0] * (self.args.pseudo_pos_num - len(candidate_confidences))

        pseudo_negative = self.negative_sampler(candidate_knowledges_pos, candidate_knowledges)
        # pseudo_negative = [] # todo: remove for rarank!!!!!!!!!!

        ### Grouping
        # group_num = min(self.args.pseudo_pos_rank, len(candidate_knowledges)) - 1
        # random_idx = sorted(random.sample(list(range(1, len(candidate_knowledges))), k=group_num))
        # candidate_knowledges_pos = [candidate_knowledges[0]] + [candidate_knowledges[idx] for idx in random_idx]
        # # candidate_confidences = [candidate_confidences[0]] + [candidate_confidences[idx] for idx in random_idx]
        # # candidate_confidences = (np.array(candidate_confidences) - min(candidate_confidences)) / (max(candidate_confidences) - min(candidate_confidences) + 1e-10)
        # candidate_confidences_pos = list(np.cumprod([0.8] * len(candidate_knowledges_pos)) / 0.8)

        # sampled_pair = sorted(random.sample(list(range(len(candidate_positives_idx))), k=2))
        # pseudo_positive = candidate_positives_idx[sampled_pair[0]]
        # pseudo_negative = candidate_positives_idx[sampled_pair[1]]

        # pseudo_positive = random.choice(candidate_positives_idx)
        # pseudo_positive = candidate_positives_idx[self.args.pseudo_pos_rank - 1]

        candidate_indice = candidate_knowledges_pos + pseudo_negative  # [candidate_positives_idx[self.args.pseudo_pos_rank]]

        candidate_knowledge_text = [self.knowledgeDB[idx] for idx in candidate_indice]
        candidate_knowledge = self.tokenizer(candidate_knowledge_text, truncation=True, padding='max_length', max_length=self.args.max_length)
        candidate_knowledge_token = candidate_knowledge.input_ids
        candidate_knowledge_mask = candidate_knowledge.attention_mask
        #
        context_batch['candidate_indice'] = candidate_indice
        context_batch['candidate_knowledge_token'] = candidate_knowledge_token
        context_batch['candidate_knowledge_mask'] = candidate_knowledge_mask

        context_batch['pseudo_targets'] = candidate_knowledges_pos  # [candidate_knowledges[0]]
        context_batch['pseudo_confidences'] = candidate_confidences_pos  # + [-1e10] * (self.args.knowledge_num - len(candidate_confidences_pos))

        context_batch['target_knowledge'] = [target_knowledge_idx]  # candidate_knowledges[:3]  # target_knowledge_idx
        context_batch['all_negative'] = candidate_knowledges + self.all_negative(candidate_knowledges)
        context_batch['bm25_top20'] = candidate_knowledges
        context_batch['new_knowledge'] = self.knowledgeDB[target_knowledge_idx] not in self.train_knowledgeDB

        context_batch['indices'] = idx
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.args.device)
                # context_batch[k] = torch.as_tensor(v)
        return context_batch

    def __len__(self):
        return len(self.augmented_raw_sample)


class KnowledgeDataset(Dataset):
    def __init__(self, args, knowledgeDB, tokenizer):
        super(Dataset, self).__init__()
        self.tokenizer = tokenizer
        self.know_max_length = args.know_max_length
        self.knowledgeDB = knowledgeDB
        self.data_samples = []

    def __getitem__(self, item):
        data = self.knowledgeDB[item]
        tokenized_data = self.tokenizer(data,
                                        max_length=self.know_max_length,
                                        padding='max_length',
                                        truncation=True,
                                        add_special_tokens=True)
        tokens = torch.LongTensor(tokenized_data.input_ids)
        mask = torch.LongTensor(tokenized_data.attention_mask)
        docid = self.tokenizer.encode(convert_idx_to_docid(item), truncation=True, padding='max_length', max_length=10)[1:-1]  # 이미 Tensor로 받음
        docid = torch.LongTensor(docid)
        return tokens, mask, docid

    def __len__(self):
        return len(self.knowledgeDB)


class KnowledgeTopicDataset(Dataset):
    def __init__(self, args, knowledgeTopicDB, tokenizer):
        super(Dataset, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.know_max_length = args.know_max_length
        self.knowledgeTopicDB = knowledgeTopicDB
        self.data_samples = []

    def __getitem__(self, item):
        topic, data = self.knowledgeTopicDB[item]
        topic_idx = self.args.topicDic['str'][topic]
        tokenized_data = self.tokenizer(data,
                                        max_length=self.know_max_length,
                                        padding='max_length',
                                        truncation=True,
                                        add_special_tokens=True)
        tokens = torch.LongTensor(tokenized_data.input_ids)
        mask = torch.LongTensor(tokenized_data.attention_mask)
        # docid = self.tokenizer.encode(convert_idx_to_docid(item), truncation=True, padding='max_length', max_length=10)[1:-1]  # 이미 Tensor로 받음
        # docid = torch.LongTensor(docid)
        return tokens, mask, topic_idx

    def __len__(self):
        return len(self.knowledgeTopicDB)


class TopicDataset(Dataset):  # knowledge용 데이터셋
    def __init__(self, args, data_sample, knowledgeDB, train_knowledgeDB, tokenizer, task, mode='train'):
        super(Dataset, self).__init__()
        self.args = args
        self.task = task
        self.tokenizer = tokenizer
        self.knowledgeDB = knowledgeDB
        self.train_knowledgeDB = train_knowledgeDB
        self.augmented_raw_sample = data_sample
        self.mode = mode

    def __getitem__(self, idx):  # TODO 구현 전
        data = self.augmented_raw_sample[idx]
        cbdicKeys = ['dialog', 'user_profile', 'response', 'type', 'topic', 'situation', 'target_knowledge', 'candidate_knowledges', 'candidate_confidences']
        dialog, user_profile, response, type, topic, situation, target_knowledge_idx, candidate_knowledges, candidate_confidences = [data[i] for i in cbdicKeys]
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        context_batch = defaultdict()
        prefix = '<profile>' + user_profile + '<type>' + type + self.tokenizer.sep_token

        prefix_encoding = self.tokenizer.encode(prefix)[1:-1][:self.args.max_prefix_length]
        input_sentence = self.tokenizer('<dialog>' + dialog, add_special_tokens=False).input_ids

        input_sentence = [self.tokenizer.cls_token_id] + prefix_encoding + input_sentence[-(self.args.max_length - len(prefix_encoding) - 1):]
        input_sentence = input_sentence + [pad_token_id] * (self.args.max_length - len(input_sentence))

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

        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.args.device)
                # context_batch[k] = torch.as_tensor(v)
        return context_batch

    def __len__(self):
        return len(self.augmented_raw_sample)
