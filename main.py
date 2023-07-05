import copy
import json
import sys
import logging
from collections import defaultdict
import random

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BartForConditionalGeneration, GPT2LMHeadModel, GPT2Config, AutoConfig, BartTokenizer
import data
from config import bert_special_tokens_dict, gpt_special_tokens_dict
from data_model import GenerationDataset, DialogDataset, KnowledgeDataset, KnowledgeTopicDataset, TopicDataset
from eval_know import eval_know, knowledge_reindexing
from train_goal_topic import train_goal_topic, write_goal_topic_result
from train_know import train_know
from train_topic import train_topic, pretrain_topic, train_goal
from utils import *
from models import *
from data_util import readDic, dataset_reader, process_augment_sample, bm_tokenizer, process_augment_sample_topic
from rank_bm25 import BM25Okapi
import nltk

nltk.download('stopwords')


# def train_knowledge_indexing(args, knowledge_data, retriever, optimizer):
#     # 모든 know_index를 버트에 태움
#     print('...train knowledge indexing...')
#     knowledgeDataLoader = DataLoader(
#         knowledge_data,
#         batch_size=args.batch_size
#     )
#     knowledge_index = []
#     criterion = nn.CrossEntropyLoss()
#     train_epoch_loss = 0
#     for batch in tqdm(knowledgeDataLoader):
#         input_ids = batch[0].to(args.device)
#         attention_mask = batch[1].to(args.device)
#         target_know_idx = batch[2].to(args.device)
#
#         if args.know_ablation == 'bart':
#             loss = retriever.knowledge_retrieve(input_ids, attention_mask, None, None, labels=target_know_idx)
#         else:
#             logit = retriever.knowledge_retrieve(input_ids, attention_mask, None, None, ablation='mlp')
#             loss = criterion(logit, target_know_idx)
#
#         train_epoch_loss += loss
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print(f"Knowledge Indexing Loss: {train_epoch_loss}")

def split_validation(train_dataset_raw, train_ratio=1.0):
    # train_set_x, train_set_y = train_set
    n_samples = len(train_dataset_raw)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * train_ratio))
    train_set = [train_dataset_raw[s] for s in sidx[:n_train]]
    valid_set = [train_dataset_raw[s] for s in sidx[n_train:]]

    return train_set, valid_set


def main():
    test = read_pkl('augmented_raw_sample_goal.txt')
    # TH 작업 main
    args = parseargs()
    # args.data_cache = False
    args.who = "TH"
    # args.bert_name = 'facebook/bart-base'
    # args.task = 'know'
    print(args)

    checkPath(args.log_dir)
    checkPath(args.model_dir)

    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(args.log_dir, f'{args.time}_{args.log_name + "_"}log.txt'), filemode='a', format='%(asctime)s: %(levelname)s: %(message)s', datefmt='%Y/%m/%d_%p_%I:%M:%S ')
    logging.info('Commend: {}'.format(', '.join(map(str, sys.argv))))
    # Model cached load
    # checkPath(os.path.join("cache", args.bert_name))
    bert_model = AutoModel.from_pretrained(args.bert_name)
    bert_config = AutoConfig.from_pretrained(args.bert_name)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
    tokenizer.add_special_tokens(bert_special_tokens_dict)  # [TH] add bert special token (<dialog>, <topic> , <type>)
    bert_model.resize_token_embeddings(len(tokenizer))
    args.hidden_size = bert_model.config.hidden_size  # BERT large 쓸 때 대비

    topicDic = readDic(os.path.join(args.data_dir, "topic2id.txt"))
    goalDic = readDic(os.path.join(args.data_dir, "goal2id.txt"))
    args.topicDic = topicDic
    args.goalDic = goalDic
    args.topic_num = len(topicDic['int'])
    args.goal_num = len(goalDic['int'])

    # modules = [bert_model.encoder.layer[:bert_config.num_hidden_layers - 2], bert_model.embeddings]
    # for module in modules:
    #     for param in module.parameters():
    #         param.requires_grad = False
    # Read knowledge DB
    # train_knowledgeDB = data.read_pkl(os.path.join(args.data_dir, 'train_knowledge_DB.pickle'))  # TODO: verbalize (TH)

    train_dataset_raw, train_knowledge_base, train_knowledge_topic = dataset_reader(args, 'train')
    test_dataset_raw, valid_knowledge_base, test_knowledge_topic = dataset_reader(args, 'test')
    valid_dataset_raw, test_knowledge_base, _ = dataset_reader(args, 'dev')

    train_knowledgeDB, all_knowledgeDB = set(), set()
    train_knowledgeDB.update(train_knowledge_base)

    all_knowledgeDB.update(train_knowledge_base)
    all_knowledgeDB.update(valid_knowledge_base)
    all_knowledgeDB.update(test_knowledge_base)

    train_knowledgeDB = list(train_knowledgeDB)
    all_knowledgeDB = list(all_knowledgeDB)

    filtered_corpus = []
    for sentence in all_knowledgeDB:
        tokenized_sentence = bm_tokenizer(sentence, tokenizer)
        # tokenized_sentence = [word for word in tokenized_sentence if word not in stop_words]
        filtered_corpus.append(tokenized_sentence)
    args.bm25 = BM25Okapi(filtered_corpus)

    # knowledgeDB = data.read_pkl(os.path.join(args.data_dir, 'knowledgeDB.txt'))  # TODO: verbalize (TH)
    # knowledgeDB.insert(0, "")
    args.train_knowledge_num = len(train_knowledgeDB)
    args.train_knowledgeDB = train_knowledgeDB

    args.all_knowledge_num = len(all_knowledgeDB)
    args.all_knowledgeDB = all_knowledgeDB

    if 'goal' in args.task:
        # KNOWLEDGE TASk
        retriever = Retriever(args, bert_model)
        retriever = retriever.to(args.device)

        # pretrain_topic(args, retriever, train_knowledge_topic, test_knowledge_topic, tokenizer)

        train_dataset = process_augment_sample_topic(train_dataset_raw, tokenizer, train_knowledgeDB)
        valid_dataset = process_augment_sample_topic(valid_dataset_raw, tokenizer, all_knowledgeDB)
        test_dataset = process_augment_sample_topic(test_dataset_raw, tokenizer, all_knowledgeDB)

        train_datamodel_topic = TopicDataset(args, train_dataset, train_knowledgeDB, train_knowledgeDB, tokenizer, task='know')
        valid_datamodel_topic = TopicDataset(args, valid_dataset, all_knowledgeDB, train_knowledgeDB, tokenizer, task='know')
        test_datamodel_topic = TopicDataset(args, test_dataset, all_knowledgeDB, train_knowledgeDB, tokenizer, task='know')

        train_dataloader_topic = DataLoader(train_datamodel_topic, batch_size=args.batch_size, shuffle=True)
        valid_dataloader_topic = DataLoader(valid_datamodel_topic, batch_size=args.batch_size, shuffle=False)
        test_dataloader_topic = DataLoader(test_datamodel_topic, batch_size=args.batch_size, shuffle=False)

        train_goal(args, retriever, train_dataloader_topic, test_dataloader_topic, tokenizer)

    if 'resp' in args.task:
        # config = GPT2Config.from_pretrained(args.bert_name, max_length=args.max_gen_length+args.max_length) # for GPT
        # gpt_model = GPT2LMHeadModel.from_pretrained(args.gpt_name, cache_dir=os.path.join("cache", args.gpt_name)) # for GPT
        # tokenizer = AutoTokenizer.from_pretrained(args.gpt_name)
        tokenizer = BartTokenizer.from_pretrained(args.bart_name)
        gpt_model = BartForConditionalGeneration.from_pretrained(args.bart_name)

        # tokenizer.pad_token = tokenizer.eos_token # for GPT
        tokenizer.add_special_tokens(gpt_special_tokens_dict)  # [TH] add bert special token (<dialog>, <topic> , <type>)

        gpt_model.resize_token_embeddings(len(tokenizer))
        args.hidden_size = gpt_model.config.hidden_size  # BERT large 쓸 때 대비

        # train_dataset_resp = process_augment_sample(train_dataset_raw, tokenizer, train_knowledgeDB)
        # test_dataset_resp = process_augment_sample(test_dataset_raw, tokenizer, all_knowledgeDB)

        train_dataset_resp = process_augment_sample_topic(train_dataset_raw, tokenizer, train_knowledgeDB)  # for topic-task (모든 시스템 발화)
        test_dataset_resp = process_augment_sample(test_dataset_raw, tokenizer, all_knowledgeDB)  # for knowledge-task (시스템 발화 중 knowledge 가 태깅되어 있는거)
        # test_dataset_resp_retrieve = process_augment_sample(test_dataset_raw, tokenizer, all_knowledgeDB)

        # with open('augmented_dataset_test.txt', 'rb') as f:
        #     test_dataset_resp = pickle.load(f)
        # for sample in test_dataset_resp:
        #     sample['dialog'] = sample['dialog'].replace('[SEP]', tokenizer.eos_token)
        #     sample['target_knowledge'] = all_knowledgeDB.index(sample['target_knowledge'])
        #     sample['candidate_knowledges'] = [all_knowledgeDB.index(c_know) for c_know in sample['candidate_knowledges']]

        train_datamodel_resp = GenerationDataset(args, train_dataset_resp, train_knowledgeDB, tokenizer, mode='train', subtask=args.subtask)
        test_datamodel_resp = GenerationDataset(args, test_dataset_resp, all_knowledgeDB, tokenizer, mode='test', subtask=args.subtask)
        # test_datamodel_resp_retrieve = GenerationDataset(args, test_dataset_resp_retrieve, all_knowledgeDB, tokenizer, mode='test', subtask=args.subtask)

        train_dataloader_resp = DataLoader(train_datamodel_resp, batch_size=args.batch_size, shuffle=True)
        test_dataloader_resp = DataLoader(test_datamodel_resp, batch_size=args.batch_size, shuffle=False)
        # test_dataloader_resp_retrieve = DataLoader(test_datamodel_resp_retrieve, batch_size=args.batch_size, shuffle=False)

        generator = Retriever(args, gpt_model=gpt_model)
        generator = generator.to(args.device)

        if args.saved_goal_model_path != '':
            generator.load_state_dict(torch.load(os.path.join(args.model_dir, f"{args.saved_goal_model_path}.pt")))
        else:
            train_goal_topic(args, generator, tokenizer, train_dataloader_resp, test_dataloader_resp, 'goal')

        augmented_raw_sample_goal = write_goal_topic_result(args, generator, tokenizer, test_dataloader_resp, 'goal')
        test_dataloader_resp.dataset.augmented_raw_sample = augmented_raw_sample_goal
        write_pkl(augmented_raw_sample_goal, "augmented_raw_sample_goal.txt")

        if args.saved_topic_model_path != '':
            generator.load_state_dict(torch.load(os.path.join(args.model_dir, f"{args.saved_topic_model_path}.pt")))
        else:
            train_goal_topic(args, generator, tokenizer, train_dataloader_resp, test_dataloader_resp, 'topic')

        augmented_raw_sample_topic = write_goal_topic_result(args, generator, tokenizer, test_dataloader_resp, 'topic')
        test_dataloader_resp.dataset.augmented_raw_sample = augmented_raw_sample_topic
        write_pkl(augmented_raw_sample_topic, "augmented_raw_sample_topic.txt")

    if 'topic' in args.task:
        # KNOWLEDGE TASk
        retriever = Retriever(args, bert_model)
        retriever = retriever.to(args.device)

        # pretrain_topic(args, retriever, train_knowledge_topic, test_knowledge_topic, tokenizer)

        train_dataset = process_augment_sample_topic(train_dataset_raw, tokenizer, train_knowledgeDB)
        valid_dataset = process_augment_sample_topic(valid_dataset_raw, tokenizer, all_knowledgeDB)
        test_dataset = process_augment_sample_topic(test_dataset_raw, tokenizer, all_knowledgeDB)

        train_datamodel_topic = TopicDataset(args, train_dataset, train_knowledgeDB, train_knowledgeDB, tokenizer, task='know')
        valid_datamodel_topic = TopicDataset(args, valid_dataset, all_knowledgeDB, train_knowledgeDB, tokenizer, task='know')
        test_datamodel_topic = TopicDataset(args, test_dataset, all_knowledgeDB, train_knowledgeDB, tokenizer, task='know')

        train_dataloader_topic = DataLoader(train_datamodel_topic, batch_size=args.batch_size, shuffle=True)
        valid_dataloader_topic = DataLoader(valid_datamodel_topic, batch_size=args.batch_size, shuffle=False)
        test_dataloader_topic = DataLoader(test_datamodel_topic, batch_size=args.batch_size, shuffle=False)

        train_topic(args, retriever, train_dataloader_topic, test_dataloader_topic, tokenizer)

    if 'know' in args.task:
        # bert_model = AutoModel.from_pretrained(args.bert_name, cache_dir=os.path.join("cache", args.bert_name))

        # KNOWLEDGE TASk
        retriever = Retriever(args, bert_model)
        retriever = retriever.to(args.device)

        train_knowledge_data = KnowledgeDataset(args, train_knowledgeDB, tokenizer)  # knowledge dataset class
        all_knowledge_data = KnowledgeDataset(args, all_knowledgeDB, tokenizer)  # knowledge dataset class

        # train_dataset_raw = dataset_reader(args, 'train')
        # test_dataset_raw = dataset_reader(args, 'test')
        goal_list = []
        # ['Movie recommendation', 'POI recommendation', 'Music recommendation', 'QA']
        if 'Movie' in args.goal_list:
            goal_list.append('Movie recommendation')
        if 'POI' in args.goal_list:
            goal_list.append('POI recommendation')
        if 'Music' in args.goal_list:
            goal_list.append('Music recommendation')
        if 'QA' in args.goal_list:
            goal_list.append('Q&A')

        train_dataset_raw, valid_dataset_raw = split_validation(train_dataset_raw, args.train_ratio)
        train_dataset = process_augment_sample(train_dataset_raw, tokenizer, train_knowledgeDB, goal_list=goal_list)
        valid_dataset = process_augment_sample(valid_dataset_raw, tokenizer, all_knowledgeDB)
        # test_dataset = process_augment_sample(test_dataset_raw, tokenizer, all_knowledgeDB)  # gold-topic
        test_dataset = read_pkl("augmented_raw_sample_topic.txt")
        for aug_data in test_dataset:
            aug_data['dialog'] = aug_data['dialog'].replace('</s>', '[SEP]')

        train_datamodel_know = DialogDataset(args, train_dataset, train_knowledgeDB, train_knowledgeDB, tokenizer, task='know')
        valid_datamodel_know = DialogDataset(args, valid_dataset, all_knowledgeDB, train_knowledgeDB, tokenizer, task='know')
        test_datamodel_know = DialogDataset(args, test_dataset, all_knowledgeDB, train_knowledgeDB, tokenizer, task='know')

        train_dataloader = DataLoader(train_datamodel_know, batch_size=args.batch_size, shuffle=True)
        train_dataloader_retrieve = DataLoader(train_datamodel_know, batch_size=args.batch_size, shuffle=False)
        valid_dataloader = DataLoader(test_datamodel_know, batch_size=args.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_datamodel_know, batch_size=1, shuffle=False)

        # print('rerank mode')
        # retriever.init_reranker()
        # train_know(args, train_dataloader, valid_dataloader, retriever, knowledge_data, knowledgeDB, tokenizer)

        # if args.saved_model_path == '':
        #     print('retrieve mode')
        #     args.stage = 'retrieve'
        #     eval_know(args, valid_dataloader, retriever, all_knowledge_data, all_knowledgeDB, tokenizer)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리
        #     train_know(args, train_dataloader, valid_dataloader, retriever, train_knowledge_data, train_knowledgeDB, all_knowledge_data, all_knowledgeDB, tokenizer)
        #
        #     # eval_know(args, train_dataloader_retrieve, retriever, train_knowledge_data, train_knowledgeDB, tokenizer, retrieve=True)  # todo: remove
        #     # eval_know(args, valid_dataloader, retriever, all_knowledge_data, all_knowledgeDB, tokenizer, retrieve=True)  # todo: remove
        #
        #     args.stage = 'rerank'
        # else:
        #     print('############################retriever load:\t%s#################################' % args.saved_model_path)
        #     retriever.load_state_dict(torch.load(os.path.join(args.model_dir, args.saved_model_path), map_location=args.device))

        if args.stage == 'rerank':
            args.stage = 'retrieve'
            # eval_know(args, valid_dataloader, retriever, all_knowledge_data, all_knowledgeDB, tokenizer)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리

            print('rerank mode')
            args.stage = 'rerank'
            retriever.init_reranker()
            # modules = [retriever.rerank_bert.encoder.layer[:bert_config.num_hidden_layers - 2], retriever.rerank_bert.embeddings]
            # for module in modules:
            #     for param in module.parameters():
            #         param.requires_grad = False

            args.lr = args.lr_rerank

            train_know(args, train_dataloader, valid_dataloader, retriever, train_knowledge_data, train_knowledgeDB, all_knowledge_data, all_knowledgeDB, tokenizer)
            # eval_know(args, test_dataloader, retriever, all_knowledge_data, all_knowledgeDB, tokenizer, write=True)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리


if __name__ == "__main__":
    main()
    # logging.info('Commend: {}'.format(', '.join(map(str, sys.argv))))
