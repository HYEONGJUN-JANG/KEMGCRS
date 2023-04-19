from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim
from data_temp import DialogDataset_TEMP
from eval_know import knowledge_reindexing, eval_know
from metric import EarlyStopping
from utils import *
from models import *
import logging

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
            # response = batch['response']
            candidate_knowledge_token = batch['candidate_knowledge_token']  # [B,5,256]
            candidate_knowledge_mask = batch['candidate_knowledge_mask']  # [B,5,256]
            pseudo_positive_idx = torch.stack([idx[0] for idx in batch['candidate_indice']])

            # target_knowledge = candidate_knowledge_token[:, 0, :]

            target_knowledge_idx = batch['target_knowledge']  # [B,5,256]

            logit = retriever.compute_know_score(dialog_token, dialog_mask, knowledge_index)

            if args.know_ablation == 'target':
                loss = criterion(logit, target_knowledge_idx)  # For MLP predict

            elif args.know_ablation == 'pseudo':
                # loss = criterion(logit, pseudo_positive_idx)  # For MLP predict
                logit = retriever.knowledge_retrieve(dialog_token, dialog_mask, candidate_knowledge_token, candidate_knowledge_mask)
                predicted_positive = logit[:, 0]  # [B]
                predicted_negative = logit[:, 1:]  # [B, K]
                relative_preference = predicted_positive.unsqueeze(1) - predicted_negative  # [B, K]
                loss = -relative_preference.sigmoid().log().sum(dim=1).mean()

            # args.loss_rec = 'bpr'
            # if args.loss_rec == 'cross_entropy':
            #     pass
            # elif args.loss_rec == 'bpr':
            #     logit = retriever.knowledge_retrieve(dialog_token, dialog_mask, candidate_knowledge_token, candidate_knowledge_mask)
            #     predicted_positive = logit[:, 0]  # [B]
            #     predicted_negative = logit[:, 1:]  # [B, K]
            #     relative_preference = predicted_positive.unsqueeze(1) - predicted_negative  # [B, K]
            #     loss = -relative_preference.sigmoid().log().mean()

            train_epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_update += 1

            if num_update > update_freq:
                update_key_bert(retriever.key_bert, retriever.query_bert)
                num_update = 0
                knowledge_index = knowledge_reindexing(args, knowledge_data, retriever)
                knowledge_index = knowledge_index.to(args.device)

        print(f"Epoch: {epoch}\nTrain Loss: {train_epoch_loss}")
        eval_know(args, test_dataloader, retriever, knowledge_data, knowledgeDB, tokenizer, knowledge_index)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리
