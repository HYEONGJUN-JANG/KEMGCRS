import sys
import logging
from transformers import AutoModel, AutoTokenizer
import data
from config import bert_special_tokens_dict
from train_goal_topic import train_topic, train_goal
from utils import *
import models
# from models_hj import *
from data_util import readDic
from platform import system as sysChecker
import dataModel
from collections import defaultdict
from train_know import train_retriever_idx

def main():
    # HJ 작업 --> 형준 여기서만 작업
    args = parseargs()
    args.data_cache = True
    args.batch_size = 4
    args.log_name = "log_Topic PRF"
    args.task = 'topic'
    args.num_epochs = 1
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
    knowledgeDB = data.read_pkl(os.path.join(args.data_dir, 'knowledgeDB.txt'))  # TODO: verbalize (TH)
    knowledge_data = dataModel.KnowledgeDataset(args, knowledgeDB, tokenizer)  # knowledge dataset class
    args.knowledge_num = len(knowledgeDB)

    all_knowledgeDB = data.read_pkl(os.path.join(args.data_dir, 'all_knowledge_DB.pickle'))  # TODO: verbalize (TH)
    knowledgeDB_values = [k[1] for k in all_knowledgeDB]
    knowledgeDB_entity_values = defaultdict(list)
    for k in all_knowledgeDB:
        knowledgeDB_entity_values[k[0]].append(knowledgeDB_values.index(k[1]))


    topicDic_str, topicDic_int = readDic(os.path.join(args.data_dir, "topic2id.txt"))
    goalDic_str, goalDic_int = readDic(os.path.join(args.data_dir, "goal2id.txt"))
    args.topic_num = len(topicDic_str)
    args.goal_num = len(goalDic_str)

    # TODO: retriever 로 바꿔서 save 와 load
    retriever = models.Retriever(args, bert_model1, bert_model2)
    retriever = retriever.to(args.device)

    # HJ Task (Type, Topic)
    args.who = "HJ"
    args.task = 'goal'
    print(f"Training {args.task} Task")
    train_dataloader = data.dataset_reader(args, tokenizer, knowledgeDB, mode='train', goal_dict=goalDic_str, topic_dict=topicDic_str)
    test_dataloader = data.dataset_reader(args, tokenizer, knowledgeDB, mode='test', goal_dict=goalDic_str, topic_dict=topicDic_str)
    train_goal(args, train_dataloader, test_dataloader, retriever, goalDic_int, tokenizer)

    args.task = 'topic'
    print(f"Training {args.task} Task")
    train_dataloader = data.dataset_reader(args, tokenizer, knowledgeDB, mode='train', goal_dict=goalDic_str, topic_dict=topicDic_str)
    test_dataloader = data.dataset_reader(args, tokenizer, knowledgeDB, mode='test', goal_dict=goalDic_str, topic_dict=topicDic_str)
    train_topic(args, train_dataloader, test_dataloader, retriever, goalDic_int, topicDic_int, tokenizer)

    # # TH Task (Know)
    args.who = "TH"
    args.task = 'know'
    print(f"Training {args.task} Task")
    args.batch_size = args.batch_size//2
    trainKnow_dataloader = data.dataset_reader(args, tokenizer, knowledgeDB, mode='train')
    testKnow_dataloader = data.dataset_reader(args, tokenizer, knowledgeDB, mode='test')
    train_retriever_idx(args, trainKnow_dataloader, knowledge_data, retriever)  # [TH] <topic> 추가됐으니까 재학습
    # # Just Fine_tune on Golden Target
    # train_goal(args, train_dataloader, test_dataloader, retriever, goalDic_int, tokenizer)
    # train_topic(args, train_dataloader, test_dataloader, retriever, goalDic_int, topicDic_int, tokenizer)
    # if args.saved_model_path == '':
    #     train_retriever_idx(args, trainKnow_dataloader, knowledge_data, retriever)  # [TH] <topic> 추가됐으니까 재학습
    # else:
    #     retriever.load_state_dict(torch.load(os.path.join(args.model_dir, args.saved_model_path)))
    # eval_know(args, testKnow_dataloader, retriever, knowledge_data, knowledgeDB, tokenizer)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리

    # Pipeline Fine-tune
    # for batch in train_dataloader:
        # batch...............
        # goalpreds=retriever.goal_selection()
        # topicpreds=retriever.topic_selection()
        # knowledges=retriever.knowledge_retrieve()
        # Generator.generate(asdfasdfasdf)




if __name__ == "__main__":
    main()
    # logging.info('Commend: {}'.format(', '.join(map(str, sys.argv))))
