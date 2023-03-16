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
from data_util import readDic

def knowledge_reindexing(args, knowledge_data, retriever):
    print('...knowledge indexing...')
    knowledgeDataLoader = DataLoader(
        knowledge_data,
        batch_size=args.batch_size
    )
    knowledge_index = []

    for batch in tqdm(knowledgeDataLoader):
        input_ids = batch[0].to(args.device)
        attention_mask = batch[1].to(args.device)
        knowledge_emb = retriever.key_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        knowledge_index.extend(knowledge_emb.cpu().detach())
    knowledge_index = torch.stack(knowledge_index, 0)
    return knowledge_index


def update_moving_average(ma_model, current_model):
    decay = 0.99
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = decay * old_weight + (1 - decay) * up_weight


def train_retriever_idx(args, train_dataloader, knowledge_data, retriever):
    # For training BERT indexing
    # train_dataloader = data_pre.dataset_reader(args, tokenizer, knowledgeDB)
    # knowledge_index = knowledge_index.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(retriever.parameters(), lr=1e-5)
    for epoch in range(args.num_epochs):
        knowledge_index = knowledge_reindexing(args, knowledge_data, retriever)
        knowledge_index = knowledge_index.to(args.device)

        total_loss = 0
        for batch in tqdm(train_dataloader):
            dialog_token, dialog_mask, target_knowledge, goal_type, response, topic = batch
            batch_size = dialog_token.size(0)
            dialog_token =dialog_token.to(args.device)
            dialog_mask = dialog_mask.to(args.device)
            target_knowledge = target_knowledge.to(args.device)
            # negative_knowledge = negative_knowledge.to(args.device)


            # tokenizer.batch_decode(dialog_token, skip_special_tokens=True)  # 'dialog context'
            # print([knowledgeDB[idx] for idx in target_knowledge]) # target knowledge

            dot_score = retriever.knowledge_retrieve(dialog_token, dialog_mask, knowledge_index)
            loss = criterion(dot_score, target_knowledge)
            total_loss += loss.data.float()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_moving_average(retriever.key_bert, retriever.query_bert)

        print('LOSS:\t%.4f' % total_loss)
    torch.save(retriever.state_dict(), os.path.join(args.model_dir, f"{args.time}_{args.model_name}_bin.pt"))  # TIME_MODELNAME 형식
    return knowledge_index

def train_retriever_mlm(args, train_dataloader, knowledge_index, retriever):
    """
    Pre-training 에서 Knowledge들을 활용해서 MLM 및 [CLS}토큰에 대한 Classification training을 진행하기 위한 code (...ing)
    :return:
    """
    # For Pre-training Retriever-BERT
    topic_idx_dict = readDic(os.path.join(args.data_dir, "topic2id.txt"), 'idx')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(retriever.parameters(), lr=1e-5)
    for epoch in range(args.num_epochs):
        total_loss = 0
        for batch in tqdm(train_dataloader): # 0: dialog_token, 1: dialog_mask, 2: target_knowledge, 3: goal_type, 4: response, 5: topic
            batch_size = batch[0].size(0)
            dialog_token = batch[0].to(args.device)
            dialog_mask = batch[1].to(args.device)
            target_knowledge = batch[2].to(args.device)
            topic = batch[5].to(args.device)
            # dot_score = retriever.topic_selection()



def eval_know(args, test_dataloader, retriever, knowledge_index, knowledgeDB, tokenizer):

    # Read knowledge DB
    # knowledge_index = knowledge_reindexing(args, knowledge_data, retriever)
    # knowledge_index = knowledge_index.to(args.device)
    jsonlineSave = []
    # bert_model = bert_model.to(args.device)

    cnt = 0
    for batch in tqdm(test_dataloader, desc="Knowledge_Test", bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'): # TODO: Knowledge task 분리중
        batch_size = batch[0].size(0)
        dialog_token = batch[0].to(args.device)
        dialog_mask = batch[1].to(args.device)
        target_knowledge = batch[2].to(args.device)
        goal_type = batch[3]  #
        response = batch[4]
        topic = batch[5]

        dot_score = retriever.knowledge_retrieve(dialog_token, dialog_mask, knowledge_index)

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
    checkPath(args.model_dir)

    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(args.log_dir, f'{args.time}_{args.log_name + "_"}log.txt'), filemode='a', format='%(asctime)s: %(levelname)s: %(message)s', datefmt='%Y/%m/%d_%p_%I:%M:%S ')
    logging.info('Commend: {}'.format(', '.join(map(str, sys.argv))))

    # Model cached load
    checkPath(os.path.join("cache", args.bert_name))
    bert_model1 = AutoModel.from_pretrained(args.bert_name, cache_dir=os.path.join("cache", args.bert_name))
    bert_model2 = AutoModel.from_pretrained(args.bert_name, cache_dir=os.path.join("cache", args.bert_name))

    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
    tokenizer.add_special_tokens(bert_special_tokens_dict)  # [TH] add bert special token (<dialog>, <topic> , <type>)
    bert_model1.resize_token_embeddings(len(tokenizer))
    bert_model2.resize_token_embeddings(len(tokenizer))
    args.hidden_size = bert_model1.config.hidden_size  # BERT large 쓸 때 대비

    # Read knowledge DB
    knowledgeDB = data.read_pkl(os.path.join(args.data_dir, args.k_DB_name))  # TODO: verbalize (TH)
    knowledge_data = KnowledgeDataset(args, knowledgeDB, tokenizer)  # knowledge dataset class

    train_dataloader = data.dataset_reader(args, tokenizer, knowledgeDB, 'train')
    test_dataloader = data.dataset_reader(args, tokenizer, knowledgeDB, 'test')
    topicDic, goalDic = readDic(os.path.join(args.data_dir, "topic2id.txt"),"idx"), readDic(os.path.join(args.data_dir, "goal2id.txt"),"idx")
    # knowledge_index = torch.tensor(np.load(os.path.join(args.data_dir, args.k_idx_name)))
    # knowledge_index = knowledge_index.to(args.device)

    # TODO: retriever 로 바꿔서 save 와 load
    # if args.model_load:
    #     bert_model.load_state_dict(torch.load(os.path.join(args.model_dir, args.pretrained_model)))  # state_dict를 불러 온 후, 모델에 저장`

    retriever = Retriever(args, bert_model1, bert_model2)
    retriever = retriever.to(args.device)


    if args.saved_model_path == '':
        knowledge_index = train_retriever_idx(args, train_dataloader, knowledge_data, retriever)  # [TH] <topic> 추가됐으니까 재학습
    else:
        retriever.load_state_dict(torch.load(os.path.join(args.model_dir, args.saved_model_path)))

    eval_know(args, test_dataloader, retriever, knowledge_index, knowledgeDB, tokenizer) # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리


if __name__ == "__main__":
    main()
    # logging.info('Commend: {}'.format(', '.join(map(str, sys.argv))))
