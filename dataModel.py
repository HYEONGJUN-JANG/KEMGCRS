import random

import torch
from torch.utils.data import Dataset


def truncationPadding(input_ids, max_length, prefix=[], suffix=[]):
    truncate_size = max_length - len(prefix) - len(suffix)
    # input_ids = prefix + input_ids[-truncate_size:] + suffix
    # input_ids = input_ids + [0] * (max_length - len(input_ids))
    # return input_ids
    if truncate_size <= len(input_ids): input_ids = prefix + input_ids[len(input_ids) - truncate_size:] + suffix
    else: input_ids = prefix + input_ids + suffix
    return input_ids + [0] * (max_length - len(input_ids))


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


class KnowDialogDataset(Dataset): # knowledge용 데이터셋
    def __init__(self, args, train_sample, knowledgeDB, tokenizer):
        super(Dataset, self).__init__()
        #self.raw_data= real_linebyline
        self.train_sample = self.process_dataset(train_sample)
        self.args = args
        self.tokenizer = tokenizer
        self.knowledgeDB = knowledgeDB
    def process_dataset(self, raw_data): return raw_data
    def __getitem__(self, idx):
        data = self.train_sample[idx]

        target_knowledge = data['target_knowledge']
        response = data['response']
        user_profile = data['user_profile']
        goal_type = data['goal_type']
        topic = data['topic']
        dialog = data['dialog']
        situation = data['situation']
        suffix = self.tokenizer.sep_token + '<type>' + goal_type + '<topic>' + topic + '<situation>' + situation
        negative_indice = self.negative_sampler(target_knowledge)
        candidate_indice = [target_knowledge] + negative_indice

        tokenized_dialog = self.tokenizer(dialog, add_special_tokens=False)
        tokenized_suffix = self.tokenizer(suffix, add_special_tokens=False)
        if self.args.input_prompt == 'dialog':
            dialog_token = truncationPadding(input_ids=tokenized_dialog.input_ids, prefix=[self.tokenizer.cls_token_id], max_length=self.args.max_length)
            dialog_mask = truncationPadding(input_ids=tokenized_dialog.attention_mask, prefix=[1], max_length=self.args.max_length)
        elif self.args.input_prompt == 'dialog_typetopic':
            dialog_token = truncationPadding(input_ids=tokenized_dialog.input_ids, prefix=[self.tokenizer.cls_token_id], suffix=tokenized_suffix.input_ids, max_length=self.args.max_length)
            dialog_mask = truncationPadding(input_ids=tokenized_dialog.attention_mask, prefix=[1],  suffix=tokenized_suffix.attention_mask, max_length=self.args.max_length)
        candidate_knowledge = self.tokenizer([self.knowledgeDB[idx] for idx in candidate_indice], truncation=True, padding='max_length', max_length=self.args.max_length)

        # target_knowledge = self.tokenizer
        candidate_knowledge_token = candidate_knowledge.input_ids
        candidate_knowledge_mask = candidate_knowledge.attention_mask

        dialog_token = torch.LongTensor(dialog_token)
        dialog_mask = torch.LongTensor(dialog_mask)
        candidate_knowledge_token = torch.LongTensor(candidate_knowledge_token)
        candidate_knowledge_mask = torch.LongTensor(candidate_knowledge_mask)

        response = self.tokenizer(response, add_special_tokens=True, max_length=self.args.max_length, padding='max_length', truncation=True)
        response_token = torch.LongTensor(response.input_ids).to(self.args.device)
        response_mask = torch.LongTensor(response.attention_mask).to(self.args.device)

        return dialog_token, dialog_mask, target_knowledge, goal_type, response_token, response_mask, topic, candidate_knowledge_token, candidate_knowledge_mask, user_profile
        # return {'dialog_token': dialog_token, 'dialog_mask': dialog_mask, 'target_knowledge': target_knowledge, 'goal_type': goal_type, 'response': response, 'topic': topic}
        # 0: dialog_token, 1: dialog_mask, 2: target_knowledge, 3: goal_type, 4: response, 5: topic

    def negative_sampler(self, target_knowledge):
        # candidate_entity = self.knowledgeDB[target_knowledge][0]
        # candiate_all_list = self.knowledgeDB_entity_values[candidate_entity]
        # negative_indice = random.choices(candiate_all_list, k=self.args.negative_num if len(candiate_all_list) > self.args.negative_num else len(candiate_all_list))
        total_knowledge_num = self.args.knowledge_num
        negative_indice = []
        while len(negative_indice) < self.args.negative_num:
            negative_idx = random.randint(0, total_knowledge_num-1)
            if (negative_idx not in negative_indice) and (negative_idx != target_knowledge):
                negative_indice.append(negative_idx)
        return negative_indice

    def __len__(self):
        return len(self.train_sample)

# class GoalDataset(Dataset)