import torch
from torch import nn


class Retriever(nn.Module):
    def __init__(self, args, bert_model):
        super(Retriever, self).__init__()
        self.bert_model = bert_model
        self.proj = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(args.hidden_size // 2, args.hidden_size)
        )

    def forward(self, token_seq, mask):
        dialog_emb = self.bert_model(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        dialog_emb = self.proj(dialog_emb)
        return dialog_emb

    def knowledge_retrieve(self, token_seq, mask, knowledge_index):
        dialog_emb = self.bert_model(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        dialog_emb = self.proj(dialog_emb)
        dot_score = torch.matmul(dialog_emb, knowledge_index.transpose(1, 0))  # [B, N]
        return dot_score


class Model(nn.Module):
    def __init__(self, bert_model, args):
        super(Model, self).__init__()
        self.args=args
        self.bert_model = bert_model