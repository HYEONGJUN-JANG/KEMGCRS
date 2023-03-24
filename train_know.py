from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim
from data_temp import DialogDataset_TEMP
from data_util import batchify
from eval_know import knowledge_reindexing
from metric import EarlyStopping
from utils import *
from models import *
import logging
logger = logging.getLogger(__name__)

def update_moving_average(ma_model, current_model):
    decay = 0.9
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = decay * old_weight + (1 - decay) * up_weight

def train_retriever_idx(args, train_dataloader, knowledge_data, retriever, tokenizer):
    assert args.task == 'know'
    logger.info("Train Retriever Index")
    # For training BERT indexing

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(retriever.parameters(), lr=args.lr)
    modelpath = os.path.join(args.model_dir, f"{args.task}_best_model.pt")
    early_stopping = EarlyStopping(patience=7, path=modelpath, verbose=True)
    if args.retrieve == 'freeze':
        knowledge_index = knowledge_reindexing(args, knowledge_data, retriever)
        knowledge_index = knowledge_index.to(args.device)
    gpucheck=True
    for epoch in range(args.num_epochs):
        print(f"[Epoch-{epoch}]")
        logger.info(f"Train Retriever Epoch: {epoch}")
        total_loss = 0
        for batch in tqdm(train_dataloader, desc="Knowledge Train", bar_format=' {l_bar} | {bar:23} {r_bar}'):

            if isinstance(train_dataloader.dataset, DialogDataset_TEMP):
                if args.task == 'know':
                    cbdicKeys = ['dialog_token', 'dialog_mask', 'response', 'type', 'topic']
                    context_batch = batchify(args, batch, tokenizer, task=args.task)
                    cbdicKeys += ['candidate_indice','candidate_knowledge_token','candidate_knowledge_mask']
                    dialog_token, dialog_mask, response, type, topic, candidate_indice, candidate_knowledge_token, candidate_knowledge_mask = [context_batch[i] for i in cbdicKeys]
                else:
                    dialog_token, dialog_mask, response, type, topic = [context_batch[i] for i in cbdicKeys]
                target_knowledge = candidate_indice[:,0]
            else: # isinstance(train_know_DataLoader.dataset, dataModel.KnowDialogDataset)
                dialog_token, dialog_mask, target_knowledge, goal_type, response, response_mask, topic, candidate_knowledge_token, candidate_knowledge_mask, user_profile = batch
                batch_size = dialog_token.size(0)
                dialog_token =dialog_token.to(args.device)
                dialog_mask = dialog_mask.to(args.device)
                target_knowledge = target_knowledge.to(args.device) #[B]
                candidate_knowledge_token = candidate_knowledge_token.to(args.device) # [B,5,256]
                candidate_knowledge_mask = candidate_knowledge_mask.to(args.device) # [B,5,256]

            if args.retrieve == 'freeze':
                dot_score = retriever.compute__know_score(dialog_token, dialog_mask, knowledge_index)
                loss = criterion(dot_score, target_knowledge)
            else:
                logit = retriever.knowledge_retrieve(dialog_token, dialog_mask, candidate_knowledge_token, candidate_knowledge_mask)
                loss = (-torch.log_softmax(logit, dim=1).select(dim=1, index=0)).mean()

            total_loss += loss.data.float()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.momentum:
                update_moving_average(retriever.key_bert, retriever.query_bert)
        print('LOSS:\t%.4f' % total_loss)
        logger.info('LOSS:\t%.4f' % total_loss)
        early_stopping(round(1000 - int(total_loss), 3), retriever)
        if early_stopping.early_stop:
            print("Early stopping")
            logger.info("Early Stopping on Epoch {}, Path: {}".format(epoch, modelpath))
            break
        if gpucheck: gpucheck = checkGPU(args, logger)
    del optimizer
    torch.cuda.empty_cache()
    # 혹시모르니까 한번 더 저장
    torch.save(retriever.state_dict(), os.path.join(args.model_dir, f"{args.time}_{args.model_name}_bin.pt"))  # TIME_MODELNAME 형식
