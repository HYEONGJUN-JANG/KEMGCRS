import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys
import logging
from transformers import AutoModel, AutoTokenizer
from torch import nn, optim
import data
from config import bert_special_tokens_dict
from dataModel import KnowledgeDataset
from utils import *
from models import *


def knowledge_reindexing(args, knowledge_data, bert_model):
    print('...knowledge indexing...')
    knowledgeDataLoader = DataLoader(
        knowledge_data,
        batch_size=args.batch_size
    )
    knowledge_index = []

    for batch in tqdm(knowledgeDataLoader):
        input_ids = batch[0].to(args.device)
        attention_mask = batch[1].to(args.device)
        knowledge_emb = bert_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        # knowledge_index.append()
        knowledge_index.extend(knowledge_emb.cpu().detach())
    knowledge_index = torch.stack(knowledge_index, 0)
    return knowledge_index


def train(args, train_dataloader, knowledge_data, bert_model):
    # For training BERT indexing
    # train_dataloader = data_pre.dataset_reader(args, tokenizer, knowledgeDB)
    # knowledge_index = knowledge_index.to(args.device)
    knowledge_index = knowledge_reindexing(args, knowledge_data, bert_model)
    knowledge_index = knowledge_index.to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(bert_model.parameters(), lr=1e-5)
    for epoch in range(args.num_epochs):


        total_loss = 0
        for batch in tqdm(train_dataloader):
            batch_size = batch[0].size(0)
            dialog_token = batch[0].to(args.device)
            dialog_mask = batch[1].to(args.device)
            target_knowledge = batch[2].to(args.device)

            # tokenizer.batch_decode(dialog_token, skip_special_tokens=True)  # 'dialog context'
            # print([knowledgeDB[idx] for idx in target_knowledge]) # target knowledge

            dialog_emb = bert_model(input_ids=dialog_token, attention_mask=dialog_mask).last_hidden_state[:, 0, :]  # [B, d]
            dot_score = torch.matmul(dialog_emb, knowledge_index.transpose(1, 0))  # [B, N]
            loss = criterion(dot_score, target_knowledge)
            total_loss += loss.data.float()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('LOSS:\t%.4f' % total_loss)
    torch.save(bert_model.state_dict(), f"{args.time}_{args.model_name}_bin.pt")  # TIME_MODELNAME 형식

def eval_know(args, test_dataloader, bert_model, tokenizer, knowledge_index):

    # Read knowledge DB
    knowledgeDB = data.read_pkl(os.path.join(args.data_dir, args.k_DB_name))  # TODO: verbalize (TH)
    knowledge_data = KnowledgeDataset(args, knowledgeDB, tokenizer)  # knowledge dataset class
    knowledge_index = knowledge_index.to(args.device)
    jsonlineSave = []
    bert_model = bert_model.to(args.device)

    cnt = 0
    for batch in tqdm(test_dataloader, desc="Knowledge_Test", bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'): # TODO: Knowledge task 분리중
        batch_size = batch[0].size(0)
        dialog_token = batch[0].to(args.device)
        dialog_mask = batch[1].to(args.device)
        target_knowledge = batch[2].to(args.device)
        goal_type = batch[3]  #
        response = batch[4]
        topic = batch[5]

        # dialog_emb = retriever(dialog_token, dialog_mask)  # [B, d]
        dialog_emb = bert_model(input_ids=dialog_token, attention_mask=dialog_mask).last_hidden_state[:, 0, :]  # [B, d]
        dot_score = torch.matmul(dialog_emb, knowledge_index.transpose(1, 0))  # [B, N]

        top_candidate = torch.topk(dot_score, k=args.know_topk, dim=1).indices  # [B, K]

        input_text = '||'.join(tokenizer.batch_decode(dialog_token, skip_special_tokens=True))
        target_knowledge_text = [knowledgeDB[idx] for idx in target_knowledge]  # target knowledge
        retrieved_knowledge_text = [knowledgeDB[idx] for idx in top_candidate[0]]  # list
        correct = target_knowledge_text[0] in retrieved_knowledge_text

        jsonlineSave.append({'goal_type': goal_type[0], 'topic': topic, 'tf': correct, 'dialog': input_text, 'target': '||'.join(target_knowledge_text), 'response': response[0], "predict5": retrieved_knowledge_text})
        cnt += 1

    # TODO HJ: 입출력 저장 args처리 필요시 args.save_know_output 에 store_true 옵션으로 만들 필요
    write_pkl(obj=jsonlineSave, filename='jsonline.pkl')  # 입출력 저장
    save_json(args, f"{args.time}_inout", jsonlineSave)

    print('done')

def main():
    args = parseargs()
    # args.data_cache = False
    checkPath(args.log_dir)
    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(args.log_dir, f'{args.time}_{args.log_name + "_"}log.txt'), filemode='a', format='%(asctime)s: %(levelname)s: %(message)s', datefmt='%Y/%m/%d_%p_%I:%M:%S ')
    logging.info('Commend: {}'.format(', '.join(map(str, sys.argv))))

    # Model cached load
    checkPath(os.path.join("cache", args.model_name))
    bert_model = AutoModel.from_pretrained(args.model_name, cache_dir=os.path.join("cache", args.model_name))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens(bert_special_tokens_dict)  # [TH] add bert special token (<dialog>, <topic> , <type>)
    bert_model.resize_token_embeddings(len(tokenizer))

    # Read knowledge DB
    knowledgeDB = data.read_pkl(os.path.join(args.data_dir, args.k_DB_name))  # TODO: verbalize (TH)
    knowledge_index = torch.tensor(np.load(os.path.join(args.data_dir, args.k_idx_name)))
    knowledge_data = KnowledgeDataset(args, knowledgeDB, tokenizer)  # knowledge dataset class

    train_dataloader = data.dataset_reader(args, tokenizer, knowledgeDB, 'train')
    test_dataloader = data.dataset_reader(args, tokenizer, knowledgeDB, 'test')

    knowledge_index = knowledge_index.to(args.device)

    if args.model_load:
        bert_model.load_state_dict(torch.load(os.path.join(args.model_dir, args.pretrained_model)))  # state_dict를 불러 온 후, 모델에 저장`
    bert_model = bert_model.to(args.device)

    train(args, train_dataloader, knowledge_data, bert_model)  # [TH] <topic> 추가됐으니까 재학습
    eval_know(args, test_dataloader, bert_model, tokenizer, knowledge_index) # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리



if __name__ == "__main__":
    main()
    # logging.info('Commend: {}'.format(', '.join(map(str, sys.argv))))
