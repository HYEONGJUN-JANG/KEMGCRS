import random

import torch
from torch.utils.data import Dataset


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
    def __init__(self, args, train_sample):
        super(Dataset, self).__init__()
        self.train_sample = train_sample
        self.args = args

    def __getitem__(self, idx):
        data = self.train_sample[idx]
        dialog_token = torch.LongTensor(data['dialog_token'])
        dialog_mask = torch.LongTensor(data['dialog_mask'])
        target_knowledge = data['target_knowledge']
        negative_knowledge = torch.LongTensor(self.negative_sampler(target_knowledge))
        response = data['response']
        goal_type = data['goal_type']
        topic = data['topic']

        return dialog_token, dialog_mask, target_knowledge, negative_knowledge, goal_type, response, topic

    def negative_sampler(self, target_knowledge):
        total_knowledge_num = self.args.knowledge_num
        negative_indice = []
        while len(negative_indice) < self.args.negative_num:
            negative_idx = random.randint(0, total_knowledge_num)
            if (negative_idx not in negative_indice) and (negative_idx != target_knowledge):
                negative_indice.append(negative_idx)
        return negative_indice

    def __len__(self):
        return len(self.train_sample)
