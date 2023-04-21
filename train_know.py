import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim
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


def train_know(args, train_dataloader, test_dataloader, retriever, knowledge_data, knowledgeDB, tokenizer):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(retriever.parameters(), lr=args.lr)

    knowledge_index = knowledge_reindexing(args, knowledge_data, retriever)
    knowledge_index = knowledge_index.to(args.device)

    best_hit = [[], [], [], []]
    eval_metric = [-1]
    result_path = f"{args.time}_{args.model_name}_result"
    with open(os.path.join('results', result_path), 'a', encoding='utf-8') as result_f:
        result_f.write(
            '\n=================================================\n')
        result_f.write(get_time_kst())
        result_f.write('\n')
        result_f.write('Argument List:' + str(sys.argv) + '\n')
        # for i, v in vars(args).items():
        #     result_f.write(f'{i}:{v} || ')
        result_f.write('\n')
        result_f.write('[EPOCH]\tHit@1\tHit@5\tHit@10\tHit@20\n')

    for epoch in range(args.num_epochs):
        if args.update_freq == -1:
            update_freq = len(train_dataloader)
        else:
            update_freq = min(len(train_dataloader), args.update_freq)

        train_epoch_loss = 0
        num_update = 0
        for batch in tqdm(train_dataloader, desc="Knowledge_Train", bar_format=' {l_bar} | {bar:23} {r_bar}'):
            retriever.train()
            dialog_token = batch['input_ids']
            dialog_mask = batch['attention_mask']
            goal_type = batch['type']
            # response = batch['response']
            candidate_knowledge_token = batch['candidate_knowledge_token']  # [B,5,256]
            candidate_knowledge_mask = batch['candidate_knowledge_mask']  # [B,5,256]
            # pseudo_positive_idx = torch.stack([idx[0] for idx in batch['candidate_indice']])
            # pseudo_positive = batch['pseudo_positive']
            # pseudo_negative = batch['pseudo_negative']
            # target_knowledge = candidate_knowledge_token[:, 0, :]

            target_knowledge_idx = batch['target_knowledge']  # [B,5,256]

            if args.know_ablation == 'target':
                logit = retriever.compute_know_score(dialog_token, dialog_mask, knowledge_index, goal_type)
                loss = criterion(logit, target_knowledge_idx)  # For MLP predict

            elif args.know_ablation == 'pseudo':
                # dialog_token = dialog_token.unsqueeze(1).repeat(1, batch['pseudo_target'].size(1), 1).view(-1, dialog_mask.size(1))  # [B, K, L] -> [B * K, L]
                # dialog_mask = dialog_mask.unsqueeze(1).repeat(1, batch['pseudo_target'].size(1), 1).view(-1, dialog_mask.size(1))  # [B, K, L] -> [B * K, L]
                logit = retriever.compute_know_score(dialog_token, dialog_mask, knowledge_index, goal_type)
                loss = 0
                for i in range(batch['pseudo_target'].size(1)):
                    pseudo_mask = torch.zeros_like(logit)
                    pseudo_target = batch['pseudo_target'][:, i]  # [B]
                    for j in range(batch['pseudo_target'].size(1)):
                        if j != i:
                            exclude = batch['pseudo_target'][:, j]
                            pseudo_mask[torch.arange(logit.size(0)), exclude] = -1e10
                    loss += criterion(logit + pseudo_mask, pseudo_target)  # For MLP predict

                # pseudo_target = batch['pseudo_target'][:, 0]  # [B * K]
                # loss = criterion(logit, pseudo_target)  # For MLP predict
                # select_mask = torch.zeros_like(logit)
                # for i in range(batch['pseudo_target'].size(1)):
                #     exclude = batch['pseudo_target'][:, i]
                #     select_mask[torch.arange(logit.size(0)), exclude] = -1e10
                # lmb = 0.5
                # for i in range(batch['pseudo_target'].size(1) - 1):
                #     pseudo_target = batch['pseudo_target'][:, i + 1]  # [B * K]
                #     exclude = batch['pseudo_target'][:, i]
                #     select_mask[torch.arange(logit.size(0)), exclude] = -1e10
                #     select_logit = logit + select_mask
                #     loss += lmb * criterion(select_logit, pseudo_target)  # For MLP predict
                #     lmb *= lmb

                # logit = retriever.knowledge_retrieve(dialog_token, dialog_mask, candidate_knowledge_token, candidate_knowledge_mask)  # [B, 2]
                # predicted_positive = logit[:, 0]
                # predicted_negative = logit[:, 1]
                # relative_preference = predicted_positive-predicted_negative
                # loss_bpr = -relative_preference.sigmoid().log().mean()
                # loss = loss + args.loss_bpr * loss_bpr

            train_epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_update += 1

            # if num_update > update_freq:
            #     update_key_bert(retriever.key_bert, retriever.query_bert)
            #     num_update = 0
            #     knowledge_index = knowledge_reindexing(args, knowledge_data, retriever)
            #     knowledge_index = knowledge_index.to(args.device)
        knowledge_index = knowledge_reindexing(args, knowledge_data, retriever)
        knowledge_index = knowledge_index.to(args.device)
        print(f"Epoch: {epoch}\nTrain Loss: {train_epoch_loss}")

        hit1, hit5, hit10, hit20 = eval_know(args, test_dataloader, retriever, knowledge_data, knowledgeDB, tokenizer, knowledge_index)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리

        with open(os.path.join('results', result_path), 'a', encoding='utf-8') as f:
            f.write(f"%d\t%.4f\t%.4f\t%.4f\t%.4f\n" % (epoch, hit1, hit5, hit10, hit20))

        if hit10 > eval_metric[0]:
            eval_metric[0] = hit1
            best_hit[0] = hit1
            best_hit[1] = hit5
            best_hit[2] = hit10
            best_hit[3] = hit20

    print(f'BEST RESULT')
    print(f"BEST Test Hit@1: {best_hit[0]}")
    print(f"BEST Test Hit@5: {best_hit[1]}")
    print(f"BEST Test Hit@10: {best_hit[2]}")
    print(f"BEST Test Hit@20: {best_hit[3]}")

    checkPath('results')
    with open(os.path.join('results', result_path), 'a', encoding='utf-8') as f:
        f.write(f"[BEST]\t%.4f\t%.4f\t%.4f\t%.4f\n" % (best_hit[0], best_hit[1], best_hit[2], best_hit[3]))
