import torch
from torch import nn


class Retriever(nn.Module):
    def __init__(self, bert_model, hidden_size):
        super(Retriever, self).__init__()
        self.bert_model = bert_model
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )

    def forward(self, token_seq, mask):
        dialog_emb = self.bert_model(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        dialog_emb = self.proj(dialog_emb)
        return dialog_emb
