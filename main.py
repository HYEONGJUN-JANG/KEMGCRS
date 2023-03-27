import sys
import logging
from transformers import AutoModel, AutoTokenizer
import data
from config import bert_special_tokens_dict
from train_goal_topic import train_topic, train_goal
from utils import *
import models
from data_util import readDic, batchify
from platform import system as sysChecker
import dataModel
import numpy as np
from train_know import train_retriever_idx
import torch
import data_temp
from tqdm import tqdm
from torch.utils.data import DataLoader
import eval_know
import metric

def main():
    # HJ 작업 --> 형준 여기서만 작업
    args = parseargs()
    # args.do_pipeline = True
    # args.ft_type, args.ft_topic, args.ft_know = True, True, True
    if sysChecker() == 'Linux':
        pass  # HJ KT-server

    args.bert_cache_name = os.path.join(args.home, "cache", args.bert_name)

    checkPath(args.log_dir)
    checkPath(args.model_dir)

    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(args.log_dir, f'{args.time}_{args.log_name + "_"}log.txt'), filemode='a', format='%(asctime)s: %(levelname)s: %(message)s', datefmt='%Y/%m/%d_%p_%I:%M:%S ')
    logging.info('Commend: {}'.format(', '.join(map(str, sys.argv))))

    # Model cached load
    checkPath(args.bert_cache_name)
    bert_model1 = AutoModel.from_pretrained(args.bert_name, cache_dir=args.bert_cache_name)
    bert_model2 = AutoModel.from_pretrained(args.bert_name, cache_dir=args.bert_cache_name)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
    tokenizer.add_special_tokens(bert_special_tokens_dict)  # [TH] add bert special token (<dialog>, <topic> , <type>)
    bert_model1.resize_token_embeddings(len(tokenizer))
    bert_model2.resize_token_embeddings(len(tokenizer))
    args.hidden_size = bert_model1.config.hidden_size  # BERT large 쓸 때 대비

    # Read knowledge DB
    knowledgeDB = [''] + data.read_pkl(os.path.join(args.data_dir, 'knowledgeDB.txt'))  # TODO: verbalize (TH)
    args.knowledgeDB = knowledgeDB
    knowledge_data = dataModel.KnowledgeDataset(args, knowledgeDB, tokenizer)  # knowledge dataset class
    args.knowledge_num = len(knowledgeDB)

    #
    # all_knowledgeDB = data.read_pkl(os.path.join(args.data_dir, 'all_knowledge_DB.pickle'))  # TODO: verbalize (TH)
    # knowledgeDB_values = [k[1] for k in all_knowledgeDB]
    # knowledgeDB_entity_values = defaultdict(list)
    # for k in all_knowledgeDB:
    #     knowledgeDB_entity_values[k[0]].append(knowledgeDB_values.index(k[1]))


    topicDic = readDic(os.path.join(args.data_dir, "topic2id.txt"))
    goalDic = readDic(os.path.join(args.data_dir, "goal2id.txt"))
    args.topicDic = topicDic
    args.goalDic = goalDic
    args.topic_num = len(topicDic['int'])
    args.goal_num = len(goalDic['int'])

    # Default Dataset (Conversation 전체와 augmented sample존재)
    conversation_train_sample = data_temp.dataset_reader_raw_temp(args, tokenizer, knowledgeDB, data_name='train')
    conversation_test_sample = data_temp.dataset_reader_raw_temp(args, tokenizer, knowledgeDB, data_name='test')
    # Type, Topic 용 datamodel 예시
    # train_type_DataModel = data_temp.DialogDataset_TEMP(args, conversation_train_sample, knowledgeDB, tokenizer, task='type', mode='train')
    # train_type_DataLoader = DataLoader(train_type_DataModel, batch_size=args.batch_size, shuffle=True)
    # test_type_DataModel = data_temp.DialogDataset_TEMP(args, conversation_test_sample, knowledgeDB, tokenizer, task='type',mode='test')
    # test_type_DataLoader = DataLoader(test_type_DataModel, batch_size=args.batch_size, shuffle=True)
    # # train_topic_DataModel = data_temp.DialogDataset_TEMP(args, conversation_train_sample, knowledgeDB, tokenizer, task='topic', mode='train')
    # # test_topic_DataModel = data_temp.DialogDataset_TEMP(args, conversation_test_sample, knowledgeDB, tokenizer, task='topic', mode='test')
    # # Know용 데이터셋 예시
    # train_know_DataModel = data_temp.DialogDataset_TEMP(args, conversation_train_sample, knowledgeDB, tokenizer, task='know', mode='train')
    # test_know_DataModel = data_temp.DialogDataset_TEMP(args, conversation_test_sample, knowledgeDB, tokenizer, task='know', mode='test')
    # train_know_DataLoader = DataLoader(train_know_DataModel, batch_size=args.batch_size, shuffle=True)
    # test_know_DataLoader = DataLoader(test_know_DataModel, batch_size=1, shuffle=False)


    # TODO: retriever 로 바꿔서 save 와 load
    retriever = models.Retriever(args, bert_model1, bert_model2)
    retriever = retriever.to(args.device)
    ################################################################################################################
    if args.do_finetune:
        # # # HJ Task (Type, Topic)
        if args.ft_type :
            print(f"Fine-tune {args.task} Task")
            args.task = 'type'
            logging.info('Fine-tune: {} Task'.format(args.task))
            train_type_DataModel = data_temp.DialogDataset_TEMP(args, conversation_train_sample, knowledgeDB, tokenizer, task=args.task, mode='train')
            test_type_DataModel = data_temp.DialogDataset_TEMP(args, conversation_test_sample, knowledgeDB, tokenizer, task=args.task, mode='test')
            train_type_DataLoader = DataLoader(train_type_DataModel, batch_size=args.batch_size, shuffle=True)
            test_type_DataLoader = DataLoader(test_type_DataModel, batch_size=args.batch_size, shuffle=True)
            train_goal(args, train_type_DataLoader, test_type_DataLoader, retriever, tokenizer)

        if args.ft_topic:
            print(f"Fine-tune {args.task} Task")
            args.task = 'topic'
            logging.info('Fine-tune: {} Task'.format(args.task))
            train_topic_DataModel = data_temp.DialogDataset_TEMP(args, conversation_train_sample, knowledgeDB, tokenizer, task=args.task, mode='train')
            test_topic_DataModel = data_temp.DialogDataset_TEMP(args, conversation_test_sample, knowledgeDB, tokenizer, task=args.task, mode='test')
            train_topic_DataLoader = DataLoader(train_topic_DataModel, batch_size=args.batch_size, shuffle=True)
            test_topic_DataLoader = DataLoader(test_topic_DataModel, batch_size=args.batch_size, shuffle=True)
            train_topic(args, train_topic_DataLoader, test_topic_DataLoader, retriever, tokenizer)

        if args.ft_know:
            # # # TH Task (Know) -- Fine_tune on Golden Target
            # args.who = "TH"
            args.task = 'know'
            logging.info('Fine-tune: {} Task'.format(args.task))
            print(f"Fine-tune {args.task} Task")
            # 32 if sysChecker() == 'Linux' else args.batch_size
            # # TH 기존
            # train_know_DataLoader = data.dataset_reader(args, tokenizer, knowledgeDB, mode='train')
            # test_know_DataLoader = data.dataset_reader(args, tokenizer, knowledgeDB, mode='test')
            # HJ New DataLoader
            train_know_DataModel = data_temp.DialogDataset_TEMP(args, conversation_train_sample, knowledgeDB, tokenizer, task='know', mode='train')
            test_know_DataModel = data_temp.DialogDataset_TEMP(args, conversation_test_sample, knowledgeDB, tokenizer, task='know', mode='test')
            train_know_DataLoader = DataLoader(train_know_DataModel, batch_size=16 if sysChecker() == 'Linux' else args.batch_size , shuffle=True)
            test_know_DataLoader = DataLoader(test_know_DataModel, batch_size=1, shuffle=False)
            train_retriever_idx(args, train_know_DataLoader, knowledge_data, retriever, tokenizer)  # [TH] <topic> 추가됐으니까 재학습
            knowledge_index = eval_know.knowledge_reindexing(args, knowledge_data, retriever).to(args.device)


    if args.do_pipeline:
        # Pipeline Fine-tune

        args.mode = 'test'
        logging.info('Pipeline')
        # train_type_DataModel = data_temp.DialogDataset_TEMP(args, conversation_train_sample, knowledgeDB, tokenizer, task='type')
        test_type_DataModel = data_temp.DialogDataset_TEMP(args, conversation_test_sample, knowledgeDB, tokenizer, task='type', mode=args.mode)
        test_pipe_DataLoader = DataLoader(test_type_DataModel, batch_size=args.batch_size, shuffle=False)

        knowledge_index = torch.tensor(np.load(os.path.join(args.data_dir, args.k_idx_name))).to(args.device)
        label_dict = {'type':[],'topic':[],'topic5':[],'know5':[]}
        pred_dict = {'type':[],'topic':[],'topic5':[],'know5':[]}
        cnt=0
        for batch in tqdm(test_pipe_DataLoader, desc="Pipeline_Test", bar_format=' {l_bar} | {bar:23} {r_bar}'): #train_goal_topic_dataloader:
            args.task = 'type'
            context_batch = batchify(args, batch, tokenizer, task=args.task)
            modelpath = os.path.join(args.model_dir, f"{args.task}_best_model.pt")
            retriever.load_state_dict(torch.load(modelpath))
            type_score = retriever.goal_selection(context_batch['dialog_token'], context_batch['dialog_mask'])
            top1type_batch = torch.topk(type_score, k=1, dim=1).indices

            label_dict['type'].extend([args.goalDic['str'][i] for i in batch['type']])
            pred_dict['type'].extend([int(i) for i in top1type_batch])
            pred_goal_text_batch = [goalDic['int'][int(i)] for i in top1type_batch]

            args.task = 'topic'
            batch['type'] = pred_goal_text_batch
            modelpath = os.path.join(args.model_dir, f"{args.task}_best_model.pt")
            retriever.load_state_dict(torch.load(modelpath))
            context_batch = batchify(args, batch, tokenizer, task=args.task)
            topic_score = retriever.topic_selection(context_batch['dialog_token'], context_batch['dialog_mask'])
            pred_topic_text_batch = [topicDic['int'][int(i)] for i in torch.topk(topic_score, k=1, dim=1).indices]
            pred_topic5_text_batch = [[topicDic['int'][int(j)] for j in i] for i in torch.topk(topic_score, k=5, dim=1).indices]

            label_dict['topic'].extend([args.topicDic['str'][i] for i in batch['topic']])
            pred_dict['topic'].extend([int(i) for i in torch.topk(topic_score, k=1, dim=1).indices])
            pred_dict['topic5'].extend([[int(j) for j in i] for i in torch.topk(topic_score, k=5, dim=1).indices])

            args.task = 'know'
            batch['topic'] = pred_topic_text_batch
            # batch['topic'] = pred_topic5_text_batch
            modelpath = os.path.join(args.model_dir, f"{args.task}_best_model.pt")
            try: retriever.load_state_dict(torch.load(modelpath))
            except: pass
            context_batch = batchify(args, batch, tokenizer, task=args.task)
            know_score = retriever.compute__know_score(context_batch['dialog_token'], context_batch['dialog_mask'], knowledge_index)

            top5_knowledge_text = [[knowledgeDB[int(j)] for j in i] for i in torch.topk(know_score, k=5, dim=1).indices]

            pred_dict['know5'].extend([[int(j) for j in i] for i in torch.topk(know_score, k=5, dim=1).indices])
            label_dict['know5'].extend([int(i) for i in batch['target_knowledge']])
            cnt+=1
            if cnt>=5: break

        ## HJ Scoring # {'type':[],'topic':[],'topic5':[],'know5':[]}
        print("Scroing")

        for t in ['type', 'topic', 'topic5','know5']:
            logging.info("Scoring")
            if t=='know' or t=='topic5':
                hit = metric.scoring(pred_dict[t], label_dict[t])
                print(f'Pipe {t} -- hit ratio: {hit}')
                logging.info('Pipe {} -- hit ratio: {}'.format(t,hit))
            else:
                p,r,f = metric.scoring(pred_dict[t], label_dict[t])
                print(f'Pipe {t} -- P/R/F: {p}/{r}/{f}')
                logging.info('Pipe {} -- P/R/F: {}/{}/{}'.format(t, p,r,f))



if __name__ == "__main__":
    main()
    # logging.info('Commend: {}'.format(', '.join(map(str, sys.argv))))
