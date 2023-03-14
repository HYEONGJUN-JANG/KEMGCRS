import torch
from torch.utils.data import Dataset


class KnowledgeDataset(Dataset):
    def __init__(self, knowledgeDB, max_length, tokenizer):
        super(Dataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
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
    def __init__(self, train_sample):
        super(Dataset, self).__init__()
        self.train_sample = train_sample

    def __getitem__(self, idx):
        data = self.train_sample[idx]
        dialog_token = torch.LongTensor(data['dialog_token'])
        dialog_mask = torch.LongTensor(data['dialog_mask'])
        target_knowledge = data['target_knowledge']
        response = data['response']
        goal_type = data['goal_type']
        topic = data['topic']
        return dialog_token, dialog_mask, target_knowledge, goal_type, response, topic

    def __len__(self):
        return len(self.train_sample)
