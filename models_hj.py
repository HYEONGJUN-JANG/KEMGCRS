import torch
from torch import nn


class Retriever(nn.Module):
    def __init__(self, args, bert_model1, bert_model2):
        super(Retriever, self).__init__()
        self.query_bert = bert_model1 # Dialog input 받기위한 BERT
        self.key_bert = bert_model2 # Knowledge text 처리를 위한 BERT

        self.proj = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(args.hidden_size // 2, args.hidden_size)
        )
        self.pred_know = nn.Linear(args.hidden_size, args.knowledge_num)

        self.goal_proj = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(args.hidden_size // 2, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.goal_num)
        )

    def forward(self, token_seq, mask):
        dialog_emb = self.bert_model(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        dialog_emb = self.proj(dialog_emb)
        return dialog_emb

    def knowledge_retrieve(self, token_seq, mask, knowledge_index):
        batch_size = mask.size(0)
        dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        # dialog_emb = self.proj(dialog_emb)
        dot_score = torch.matmul(dialog_emb, knowledge_index.transpose(1, 0))  # [B, N]
        # knowledge_indice = torch.cat([target_knowledge.view(batch_size, -1), negative_knowledge], dim=1)
        # dot_score = self.pred_know(dialog_emb)
        return dot_score

    def goal_selection(self, token_seq, mask):
        dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        dialog_emb = self.goal_proj(dialog_emb)
        # dot_score = torch.matmul(dialog_emb, goal_idx.transpose(1,0)) #[B, N_goal]
        return dialog_emb


class Model(nn.Module):
    def __init__(self, bert_model, args):
        super(Model, self).__init__()
        self.args=args
        self.bert_model = bert_model