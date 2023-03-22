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
from collections import defaultdict
from train_know import train_retriever_idx
import torch
import data_temp
from tqdm import tqdm
from torch.utils.data import DataLoader


def main():
    # HJ 작업 --> 형준 여기서만 작업
    args = parseargs()
    args.data_cache = True
    args.batch_size = 4
    args.log_name = "log_Topic PRF"
    args.task = 'goal'
    args.num_epochs = 2
    args.do_finetune = True
    # args.do_pipeline = True
    if sysChecker() == 'Linux':
        args.home = '/home/work/CRSTEST/KEMGCRS'
        args.data_dir = os.path.join(args.home,'data')
        args.output_dir =  os.path.join(args.data_dir,'output')
        args.log_dir =  os.path.join(args.home,'logs')
        args.model_dir =  os.path.join(args.home,'models')
        args.bert_saved_model_path = os.path.join(args.home, "cache", args.bert_name)
        args.batch_size = 128
        args.num_epochs = 25
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


    all_knowledgeDB = data.read_pkl(os.path.join(args.data_dir, 'all_knowledge_DB.pickle'))  # TODO: verbalize (TH)
    knowledgeDB_values = [k[1] for k in all_knowledgeDB]
    knowledgeDB_entity_values = defaultdict(list)
    for k in all_knowledgeDB:
        knowledgeDB_entity_values[k[0]].append(knowledgeDB_values.index(k[1]))


    topicDic = readDic(os.path.join(args.data_dir, "topic2id.txt"))
    goalDic = readDic(os.path.join(args.data_dir, "goal2id.txt"))
    args.topicDic = topicDic
    args.goalDic = goalDic
    args.topic_num = len(topicDic['int'])
    args.goal_num = len(goalDic['int'])

    ###### TODO : TEMPTEMPTEMPTEMP
    # Default Dataset (Conversation 전체와 augmented sample존재)
    conversation_train_sample = data_temp.dataset_reader_raw_temp(args, tokenizer, knowledgeDB, data_name='train')
    conversation_test_sample = data_temp.dataset_reader_raw_temp(args, tokenizer, knowledgeDB, data_name='test')

    # train_type_DataModel = data_temp.DialogDataset_TEMP(args, conversation_train_sample, knowledgeDB, tokenizer, task='type', mode='train')
    # train_type_DataLoader = DataLoader(train_type_DataModel, batch_size=args.batch_size, shuffle=True)
    # test_type_DataModel = data_temp.DialogDataset_TEMP(args, conversation_test_sample, knowledgeDB, tokenizer, task='type',mode='test')
    # test_type_DataLoader = DataLoader(test_type_DataModel, batch_size=args.batch_size, shuffle=True)
    #
    # # train_topic_DataModel = data_temp.DialogDataset_TEMP(args, conversation_train_sample, knowledgeDB, tokenizer, task='topic', mode='train')
    # # test_topic_DataModel = data_temp.DialogDataset_TEMP(args, conversation_test_sample, knowledgeDB, tokenizer, task='topic', mode='test')
    # # Know용 데이터셋 예시
    # train_know_DataModel = data_temp.DialogDataset_TEMP(args, conversation_train_sample, knowledgeDB, tokenizer, task='know', mode='train')
    # test_know_DataModel = data_temp.DialogDataset_TEMP(args, conversation_test_sample, knowledgeDB, tokenizer, task='know', mode='test')
    # train_know_DataLoader = DataLoader(train_know_DataModel, batch_size=args.batch_size, shuffle=True)
    # test_know_DataLoader = DataLoader(test_know_DataModel, batch_size=1, shuffle=False)

    # TODO : type, topic , get_item 에서 dialog, type, topic, situation, profile, 각각 batchify

    #
    # task = 'know'
    # for batch in tqdm(train_know_DataLoader, desc=f"{task}_Test", bar_format=' {l_bar} | {bar:23} {r_bar}'):
    #     cbdicKeys=['dialog_token', 'dialog_mask', 'response', 'type', 'topic']
    #     if task=='know': cbdicKeys+=['candidate_indice']
    #     context_batch = batchify(args, batch, tokenizer, task='know')
    #     dialog_token, dialog_mask, response, type, topic, candidate_indice = [context_batch[i] for i in cbdicKeys]


    ## Pipeline
    # for batch in tqdm(train_know_DataLoader, desc="Know_Test", bar_format=' {l_bar} | {bar:23} {r_bar}'):
    #     batch['type'] = new_pred_goal
    #     batch['topic'] = new_pred_topic


    # TODO: retriever 로 바꿔서 save 와 load
    retriever = models.Retriever(args, bert_model1, bert_model2)
    retriever = retriever.to(args.device)
    ################################################################################################################
    if args.do_finetune:
        # # HJ Task (Type, Topic)
        print(f"Training {args.task} Task")
        args.task = 'type'
        train_type_DataModel = data_temp.DialogDataset_TEMP(args, conversation_train_sample, knowledgeDB, tokenizer, task=args.task, mode='train')
        test_type_DataModel = data_temp.DialogDataset_TEMP(args, conversation_test_sample, knowledgeDB, tokenizer, task=args.task, mode='test')
        train_type_DataLoader = DataLoader(train_type_DataModel, batch_size=args.batch_size, shuffle=True)
        test_type_DataLoader = DataLoader(test_type_DataModel, batch_size=args.batch_size, shuffle=True)
        train_goal(args, train_type_DataLoader, test_type_DataLoader, retriever, tokenizer)

        print(f"Training {args.task} Task")
        args.task = 'topic'
        train_topic_DataModel = data_temp.DialogDataset_TEMP(args, conversation_train_sample, knowledgeDB, tokenizer, task=args.task, mode='train')
        test_topic_DataModel = data_temp.DialogDataset_TEMP(args, conversation_test_sample, knowledgeDB, tokenizer, task=args.task, mode='test')
        train_topic_DataLoader = DataLoader(train_topic_DataModel, batch_size=args.batch_size, shuffle=True)
        test_topic_DataLoader = DataLoader(test_topic_DataModel, batch_size=args.batch_size, shuffle=True)
        train_topic(args, train_topic_DataLoader, test_topic_DataLoader, retriever, tokenizer)
        #
        # # # TH Task (Know) -- Fine_tune on Golden Target
        # args.who = "TH"
        # args.task = 'know'
        # print(f"Training {args.task} Task")
        # args.batch_size = args.batch_size//2
        # trainKnow_dataloader = data.dataset_reader(args, tokenizer, knowledgeDB, mode='train')
        # testKnow_dataloader = data.dataset_reader(args, tokenizer, knowledgeDB, mode='test')
        # train_retriever_idx(args, trainKnow_dataloader, knowledge_data, retriever)  # [TH] <topic> 추가됐으니까 재학습

    # # Just Fine_tune on Golden Target
    # train_goal(args, train_dataloader, test_dataloader, retriever, goalDic_int, tokenizer)
    # train_topic(args, train_dataloader, test_dataloader, retriever, goalDic_int, topicDic_int, tokenizer)
    # if args.saved_model_path == '':
    #     train_retriever_idx(args, trainKnow_dataloader, knowledge_data, retriever)  # [TH] <topic> 추가됐으니까 재학습
    # else:
    #     retriever.load_state_dict(torch.load(os.path.join(args.model_dir, args.saved_model_path)))
    # eval_know(args, testKnow_dataloader, retriever, knowledge_data, knowledgeDB, tokenizer)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리

    if args.do_pipeline:
        # Pipeline Fine-tune

        args.mode = 'test'
        # train_type_DataModel = data_temp.DialogDataset_TEMP(args, conversation_train_sample, knowledgeDB, tokenizer, task='type')
        test_type_DataModel = data_temp.DialogDataset_TEMP(args, conversation_test_sample, knowledgeDB, tokenizer, task='type')
        train_pipe_DataLoader = DataLoader(test_type_DataModel, batch_size=args.batch_size, shuffle=True)

        for batch in tqdm(train_pipe_DataLoader, desc="Topic_Test", bar_format=' {l_bar} | {bar:23} {r_bar}'): #train_goal_topic_dataloader:
            # batch
            batch_size = batch['dialog_token'].size(0)
            dialog_token = batch['dialog_token'].to(args.device)
            dialog_mask = batch['dialog_mask'].to(args.device)
            response = batch['response_token']
            golden_goal = batch['goal_type']
            golden_topic = batch['topic']
            # targets = torch.LongTensor(golden_topic).to(args.device)
            # test_label = list(map(int, golden_topic))
            ### TODO 수도코드
            type_score = retriever.goal_selection(dialog_token, dialog_mask)
            pred_goal_batch = [int(i) for i in torch.topk(type_score, k=1, dim=1).indices]
            # topic용 dialog_token 입력 처리해줘야하는부분
            topic_score = retriever.topic_selection(dialog_token, dialog_mask)# [B]
            pred_topic_batch = [int(i) for i in torch.topk(topic_score, k=1, dim=1).indices]
            # pred_dict[task][args.mode].extend(pred)
            break


                    # goalpreds.topk()
                    # topicpreds=retriever.topic_selection()
                    # topicpreds.topk()
                    # knowledges=retriever.knowledge_retrieve()
                    # knowledges.topk()
                    # Generator.generate(asdfasdfasdf)




if __name__ == "__main__":
    main()
    # logging.info('Commend: {}'.format(', '.join(map(str, sys.argv))))
