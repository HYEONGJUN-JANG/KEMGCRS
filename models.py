import torch
from torch import nn


class Retriever(nn.Module):
    def __init__(self, args, bert_model1, bert_model2):
        super(Retriever, self).__init__()
        self.args = args
        self.query_bert = bert_model1 # Dialog input 받기위한 BERT
        self.key_bert = bert_model2 # Knowledge text 처리를 위한 BERT

        self.proj = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(args.hidden_size // 2, args.hidden_size)
        )
        self.pred_know = nn.Linear(args.hidden_size, args.knowledge_num)
        self.topic_proj = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(args.hidden_size // 2, args.hidden_size)
        )


    def forward(self, token_seq, mask):
        dialog_emb = self.bert_model(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        dialog_emb = self.proj(dialog_emb)
        return dialog_emb

    def knowledge_retrieve(self, token_seq, mask, candidate_knowledge_token, candidate_knowledge_mask):
        """
        Args:
            token_seq: [B, L]
            mask: [B, L]
            candidate_knowledge_token: [B, K+1, L]
            candidate_knowledge_mask: [B, K+1, L]

        Returns:
        """

        batch_size = mask.size(0)
        dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        # dialog_emb = self.proj(dialog_emb)
        candidate_knowledge_token = candidate_knowledge_token.view(-1, self.args.max_length)  # [B, KL]
        candidate_knowledge_mask = candidate_knowledge_mask.view(-1, self.args.max_length)  # [B, KL]

        knowledge_index = self.query_bert(input_ids=candidate_knowledge_token, attention_mask=candidate_knowledge_mask).last_hidden_state[:, 0, :]
        knowledge_index = knowledge_index.view(batch_size, -1, dialog_emb.size(-1))
        logit = torch.sum(dialog_emb.unsqueeze(1) * knowledge_index, dim=2) # [B, 1, d] * [B, K+1, d] = [B, K+1, d]
        # dot_score = torch.matmul(dialog_emb, knowledge_index.transpose(1, 0))  # [B, N]
        # knowledge_indice = torch.cat([target_knowledge.view(batch_size, -1), negative_knowledge], dim=1)
        # dot_score = self.pred_know(dialog_emb)

        return logit

    def compute_score(self, token_seq, mask, knowledge_index):
        dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        dot_score = torch.matmul(dialog_emb, knowledge_index.transpose(1, 0))  # [B, N]
        return dot_score

    def topic_selection(self, token_seq, mask, topic_idx):
        dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        dialog_emb = self.topic_proj(dialog_emb)
        dot_score = torch.matmul(dialog_emb, topic_idx.transpose(1,0)) #[B, N_topic]
        return dot_score


class Model(nn.Module):
    def __init__(self, bert_model, args):
        super(Model, self).__init__()
        self.args=args
        self.bert_model = bert_model