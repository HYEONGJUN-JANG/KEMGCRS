import random

import torch
from torch.utils.data import Dataset


def truncationPadding(input_ids, max_length, prefix=[], suffix=[]):
    truncate_size = max_length - len(prefix) - len(suffix)
    input_ids = prefix + input_ids[-truncate_size:] + suffix
    input_ids = input_ids + [0] * (max_length - len(input_ids))
    return input_ids

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


class DialogDataset(Dataset):
    def __init__(self, args, train_sample, knowledgeDB, tokenizer):
        super(Dataset, self).__init__()
        self.train_sample = train_sample
        self.args = args
        self.tokenizer = tokenizer
        self.knowledgeDB = knowledgeDB

    def __getitem__(self, idx):
        data = self.train_sample[idx]

        target_knowledge = data['target_knowledge']
        response = data['response']
        goal_type = data['goal_type']
        topic = data['topic']
        dialog = data['dialog']
        suffix = '<type>' + goal_type + '<topic>' + topic
        negative_indice = self.negative_sampler(target_knowledge)
        candidate_indice = [target_knowledge] + negative_indice

        tokenized_dialog = self.tokenizer(dialog, add_special_tokens=False, max_length=self.args.max_length)
        tokenized_prefix = self.tokenizer(suffix, add_special_tokens=False, max_length=self.args.max_length)
        dialog_token = truncationPadding(input_ids=tokenized_dialog.input_ids, suffix=[self.tokenizer.cls_token_id], prefix=tokenized_prefix.input_ids, max_length=self.args.max_length)
        dialog_mask = truncationPadding(input_ids=tokenized_dialog.attention_mask, suffix=[1],  prefix=tokenized_prefix.attention_mask,max_length=self.args.max_length)
        candidate_knowledge = self.tokenizer([self.knowledgeDB[idx] for idx in candidate_indice], truncation=True, padding='max_length', max_length=self.args.max_length)
        # target_knowledge = self.tokenizer

        candidate_knowledge_token = candidate_knowledge.input_ids
        candidate_knowledge_mask = candidate_knowledge.attention_mask

        dialog_token = torch.LongTensor(dialog_token)
        dialog_mask = torch.LongTensor(dialog_mask)
        candidate_knowledge_token = torch.LongTensor(candidate_knowledge_token)
        candidate_knowledge_mask = torch.LongTensor(candidate_knowledge_mask)

        return dialog_token, dialog_mask, target_knowledge, goal_type, response, topic, candidate_knowledge_token, candidate_knowledge_mask
        # return {'dialog_token': dialog_token, 'dialog_mask': dialog_mask, 'target_knowledge': target_knowledge, 'goal_type': goal_type, 'response': response, 'topic': topic}
        # 0: dialog_token, 1: dialog_mask, 2: target_knowledge, 3: goal_type, 4: response, 5: topic

    def negative_sampler(self, target_knowledge):
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