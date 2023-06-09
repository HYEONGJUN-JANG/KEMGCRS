import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim

from data_model import KnowledgeTopicDataset
from data_temp import DialogDataset_TEMP
from eval_know import knowledge_reindexing, eval_know
from metric import EarlyStopping
from utils import *
from models import *
import logging
import numpy as np

logger = logging.getLogger(__name__)


def update_key_bert(key_bert, query_bert):
    print('update moving average')
    decay = 0  # If 0 then change whole parameter
    for current_params, ma_params in zip(query_bert.parameters(), key_bert.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = decay * old_weight + (1 - decay) * up_weight


def train_topic(args, retriever, train_dataloader_topic, test_dataloader_topic, tokenizer):
    optimizer = optim.AdamW(retriever.parameters(), lr=args.lr)
    for epoch in range(args.epoch_pt):
        total_loss = 0
        for batch in tqdm(train_dataloader_topic, bar_format=' {l_bar} | {bar:23} {r_bar}'):

            retriever.train()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            topic_idx = batch['topic_idx']

            # knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]
            knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]
            logit_topic = retriever.topic_proj(knowledge_emb)
            loss = torch.nn.CrossEntropyLoss()(logit_topic, topic_idx)

            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('LOSS_TOPIC:\t%.4f' % total_loss)

        hit1 = []
        for batch in tqdm(test_dataloader_topic, bar_format=' {l_bar} | {bar:23} {r_bar}'):
            retriever.eval()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            topic_idx = batch['topic_idx']

            # knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]
            knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]
            scores = retriever.topic_proj(knowledge_emb)

            for idx, (score, target) in enumerate(zip(scores, topic_idx)):
                top_candidate = torch.topk(score, k=1).indices
                correct_k = target in top_candidate
                hit1.append(correct_k)

        hit1 = np.average(hit1)
        print("Pre-Test Hit@1: %.4f" % np.average(hit1))


def pretrain_topic(args, retriever, train_knowledge_topic, test_knowledge_topic, tokenizer):
    train_knowledge_topic_data = KnowledgeTopicDataset(args, train_knowledge_topic, tokenizer)
    knowledgeTopicDataLoader = DataLoader(
        train_knowledge_topic_data,
        batch_size=args.batch_size
    )
    test_knowledge_topic_data = KnowledgeTopicDataset(args, test_knowledge_topic, tokenizer)
    test_knowledgeTopicDataLoader = DataLoader(
        test_knowledge_topic_data,
        batch_size=args.batch_size
    )

    optimizer = optim.AdamW(retriever.parameters(), lr=args.lr)
    for epoch in range(args.epoch_pt):
        total_loss = 0
        for batch in tqdm(knowledgeTopicDataLoader, bar_format=' {l_bar} | {bar:23} {r_bar}'):
            input_ids = batch[0].to(args.device)
            attention_mask = batch[1].to(args.device)
            topic_idx = batch[2].to(args.device)

            # knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]
            knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]
            logit_topic = retriever.topic_proj(knowledge_emb)
            loss = torch.nn.CrossEntropyLoss()(logit_topic, topic_idx)

            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('LOSS_PRETRAIN:\t%.4f' % total_loss)

        hit1 = []
        for batch in tqdm(test_knowledgeTopicDataLoader, bar_format=' {l_bar} | {bar:23} {r_bar}'):
            input_ids = batch[0].to(args.device)
            attention_mask = batch[1].to(args.device)
            topic_idx = batch[2].to(args.device)

            # knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]
            knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]
            scores = retriever.topic_proj(knowledge_emb)

            for idx, (score, target) in enumerate(zip(scores, topic_idx)):
                top_candidate = torch.topk(score, k=1).indices
                correct_k = target in top_candidate
                hit1.append(correct_k)

        hit1 = np.average(hit1)
        print("Pre-Test Hit@1: %.4f" % np.average(hit1))
