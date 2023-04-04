import torch
from torch import nn


class Retriever(nn.Module):
    def __init__(self, args, kencoder, qencoder):
        super(Retriever, self).__init__()
        self.args = args
        self.key_bert = kencoder # Knowledge text 처리를 위한 BERT
        self.query_bert = qencoder # Dialog input 받기위한 BERT
        self.q_hidden_size = qencoder.config.hidden_size
        self.k_hidden_size = kencoder.config.hidden_size

        self.proj = nn.Sequential(
            nn.Linear(self.q_hidden_size, self.q_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.q_hidden_size // 2, self.q_hidden_size)
        )
        # self.pred_know = nn.Linear(self.q_hidden_size, args.knowledge_num)

        self.goal_proj = nn.Sequential(
            nn.Linear(self.q_hidden_size, self.q_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.q_hidden_size // 2, self.q_hidden_size),
            nn.ReLU(),
            nn.Linear(self.q_hidden_size, args.goal_num)
        )
        self.topic_proj = nn.Sequential(
            nn.Linear(self.q_hidden_size, self.q_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.q_hidden_size // 2, self.q_hidden_size),
            nn.ReLU(),
            nn.Linear(self.q_hidden_size, args.topic_num)
        )


    def forward(self, token_seq, mask):
        if self.args.usebart: dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask, output_hidden_states=True).decoder_hidden_states[-1][:,0,:].squeeze(1)
        else: dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        dialog_emb = self.proj(dialog_emb)
        return dialog_emb

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
        if self.args.usebart: dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask, output_hidden_states=True).decoder_hidden_states[-1][:,0,:].squeeze(1)
        else: dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]

        # dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        candidate_knowledge_token = candidate_knowledge_token.view(-1, self.args.max_length)  # [B*(K+1), L]
        candidate_knowledge_mask = candidate_knowledge_mask.view(-1, self.args.max_length)  # [B*(K+1), L]

        knowledge_index = self.query_bert(input_ids=candidate_knowledge_token, attention_mask=candidate_knowledge_mask).last_hidden_state[:, 0, :]
        knowledge_index = knowledge_index.view(batch_size, -1, dialog_emb.size(-1))
        logit = torch.sum(dialog_emb.unsqueeze(1) * knowledge_index, dim=2)  # [B, 1, d] * [B, K+1, d] = [B, K+1]

        return logit

    def compute__know_score(self, token_seq, mask, knowledge_index):
        """
        eval_know.computing_score에서
        모든 key vector에서 올라온 벡터를 통해 계산처리
        """
        if self.args.usebart: dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask, output_hidden_states=True).decoder_hidden_states[-1][:,0,:].squeeze(1)
        else: dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        # dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        dot_score = torch.matmul(dialog_emb, knowledge_index.transpose(1, 0))  # [B, N]
        return dot_score

    def goal_selection(self, token_seq, mask, gen_labels=None): # TODO: 생성 Loss 만들기
        if self.args.usebart:
            outputs = self.query_bert(input_ids=token_seq, attention_mask=mask, output_hidden_states=True, labels=gen_labels)
            dialog_emb = outputs.decoder_hidden_states[-1][:,0,:].squeeze(1)
            gen_loss = outputs.loss
        else: # Query_bert == BERT
            dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        dialog_emb = self.goal_proj(dialog_emb)
        # dot_score = torch.matmul(dialog_emb, goal_idx.transpose(1,0)) #[B, N_goal]
        if self.args.usebart: return gen_loss, dialog_emb
        else: return dialog_emb
    def topic_selection(self, token_seq, mask, gen_labels=None):
        if self.args.usebart:
            outputs = self.query_bert(input_ids=token_seq, attention_mask=mask, output_hidden_states=True, labels=gen_labels)
            dialog_emb = outputs.decoder_hidden_states[-1][:,0,:].squeeze(1)
            gen_loss = outputs.loss
        else:
            dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]        # dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        dialog_emb = self.topic_proj(dialog_emb)
        # dot_score = torch.matmul(dialog_emb, goal_idx.transpose(1,0)) #[B, N_goal]
        # return gen_loss, dialog_emb if self.args.usebart else dialog_emb
        if self.args.usebart: return gen_loss, dialog_emb
        else: return dialog_emb



class Model(nn.Module):
    def __init__(self, bert_model, args):
        super(Model, self).__init__()
        self.args=args
        self.bert_model = bert_model