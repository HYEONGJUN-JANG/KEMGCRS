from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_util import bm_tokenizer
from utils import write_pkl, save_json
import numpy as np


def knowledge_reindexing(args, knowledge_data, retriever, stage):
    # 모든 know_index를 버트에 태움
    print('...knowledge indexing...(%s)' % stage)
    retriever.eval()
    knowledgeDataLoader = DataLoader(
        knowledge_data,
        batch_size=args.batch_size
    )
    knowledge_index = []

    for batch in tqdm(knowledgeDataLoader, bar_format=' {l_bar} | {bar:23} {r_bar}'):
        input_ids = batch[0].to(args.device)
        attention_mask = batch[1].to(args.device)
        # knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]

        if stage == 'retrieve':
            knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]
        elif stage == 'rerank':
            knowledge_emb = retriever.rerank_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]
           # logit = retriever.knowledge_retrieve(dialog_token, dialog_mask, candidate_knowledge_token, candidate_knowledge_mask)  # [B, 2]

        # knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B, d]
        # knowledge_emb = torch.sum(knowledge_emb * attention_mask.unsqueeze(-1), dim=1) / (torch.sum(attention_mask, dim=1, keepdim=True) + 1e-20)  # [B, d]

        knowledge_index.extend(knowledge_emb.cpu().detach())
    knowledge_index = torch.stack(knowledge_index, 0)
    return knowledge_index


def eval_know(args, test_dataloader, retriever, knowledge_data, knowledgeDB, tokenizer, write=None):
    retriever.eval()
    # Read knowledge DB
    # knowledge_index = knowledge_reindexing(args, knowledge_data, retriever)
    # knowledge_index = knowledge_index.to(args.device)
    jsonlineSave = []
    # bert_model = bert_model.to(args.device)

    knowledge_index = knowledge_reindexing(args, knowledge_data, retriever, stage='retrieve')
    knowledge_index = knowledge_index.to(args.device)

    # if args.stage == 'rerank':
    #     knowledge_index_rerank = knowledge_reindexing(args, knowledge_data, retriever, stage='rerank')
    #     knowledge_index_rerank = knowledge_index_rerank.to(args.device)

    goal_list = ['Movie recommendation', 'POI recommendation', 'Music recommendation', 'Q&A', 'Chat about stars']
    hit1_goal, hit5_goal, hit10_goal, hit20_goal = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    hit1, hit5, hit10, hit20 = [], [], [], []

    cnt = 0

    pred = []
    targets = []
    for batch in tqdm(test_dataloader, desc="Knowledge_Test", bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):  # TODO: Knowledge task 분리중
        dialog_token = batch['input_ids']
        dialog_mask = batch['attention_mask']
        response = batch['response']
        candidate_knowledge_token = batch['candidate_knowledge_token']  # [B,5,256]
        candidate_knowledge_mask = batch['candidate_knowledge_mask']  # [B,5,256]

        type_idx = [args.goalDic['int'][int(idx)] for idx in batch['type']]
        topic_idx = [args.topicDic['int'][int(idx)] for idx in batch['topic_idx']]

        # candidate_knowledge_mask = batch['candidate_knowledge_mask']  # [B,5,256]
        target_knowledge_idx = batch['target_knowledge']
        dot_score = retriever.compute_know_score(dialog_token, dialog_mask, knowledge_index, batch['type'])

        if args.stage == 'rerank':
            # candidate_indice = torch.topk(dot_score, k=args.know_topk, dim=1).indices  # [B, K]
            # # dot_score = retriever.compute_know_score_candidate(dialog_token, dialog_mask, knowledge_index_rerank[candidate_indice])
            # # candidate_knowledge_text = [args.knowledgeDB[idx] for idx in candidate_indice[0]]
            # candidate_knowledge_text = [args.knowledgeDB[idx] for candidates in candidate_indice for idx in candidates]
            # candidate_knowledge = tokenizer(candidate_knowledge_text, truncation=True, padding='max_length', max_length=args.max_length)
            # candidate_knowledge_token = candidate_knowledge.input_ids
            # candidate_knowledge_mask = candidate_knowledge.attention_mask
            # candidate_knowledge_token = torch.LongTensor(candidate_knowledge_token).to(args.device).view(-1, args.know_topk, args.max_length)
            # candidate_knowledge_mask = torch.LongTensor(candidate_knowledge_mask).to(args.device).view(-1, args.know_topk, args.max_length)
            dot_score = retriever.knowledge_retrieve(dialog_token, dialog_mask, candidate_knowledge_token, candidate_knowledge_mask)  # [B, 2]

        if write:
            top_candidate = torch.topk(dot_score, k=args.know_topk, dim=1).indices  # [B, K]
            input_text = '||'.join(tokenizer.batch_decode(dialog_token, skip_special_tokens=True))
            target_knowledge_text = args.knowledgeDB[target_knowledge_idx]
            retrieved_knowledge_text = [knowledgeDB[idx].lower() for idx in top_candidate[0]]  # list
            correct = target_knowledge_idx in top_candidate

            response = '||'.join(tokenizer.batch_decode(response, skip_special_tokens=True))
            query = topic_idx[0] + "|" + response
            bm_scores = args.bm25.get_scores(bm_tokenizer(query, tokenizer))
            retrieved_knowledge_score = bm_scores[top_candidate[0].cpu().numpy()]
            jsonlineSave.append({'goal_type': type_idx[0], 'topic': topic_idx[0], 'tf': correct, 'dialog': input_text, 'target': target_knowledge_text, 'response': response, "predict5": retrieved_knowledge_text, "score5": retrieved_knowledge_score})
            # save_json(args, f"{args.time}_{args.model_name}_inout", jsonlineSave)

        for idx, (score, target, goal) in enumerate(zip(dot_score, target_knowledge_idx, type_idx)):
            # goal = args.goalDic['int'][int(goal_idx)]
            # top_candidate = torch.topk(score, k=args.know_topk, dim=0).indices  # [K]
            # candidate_knowledge_text = [args.knowledgeDB[int(idx)] for idx in top_candidate]  # [K, .]
            # candidate_knowledge = tokenizer(candidate_knowledge_text, truncation=True, padding='max_length', max_length=args.max_length, return_tensors='pt')
            # candidate_knowledge_token = candidate_knowledge.input_ids.to(args.device)  # [K, L]
            # candidate_knowledge_mask = candidate_knowledge.attention_mask.to(args.device)  # [K, L]
            # re_rank_score = retriever.knowledge_retrieve(dialog_token[idx].unsqueeze(0), dialog_mask[idx].unsqueeze(0), candidate_knowledge_token.unsqueeze(0), candidate_knowledge_mask.unsqueeze(0)).squeeze(0)  # [K]

            if goal == 'Movie recommendation' or goal == 'POI recommendation' or goal == 'Music recommendation' or goal == 'Q&A':  # or goal == 'Chat about stars':
                for k in [1, 5, 10, 20]:
                    if args.stage == 'retrieve':
                        top_candidate = torch.topk(score, k=k).indices
                    elif args.stage == 'rerank':
                        top_candidate_k_idx = torch.topk(score, k=k).indices  # [B, K]
                        top_candidate = torch.gather(candidate_indice[idx], 0, top_candidate_k_idx)

                    correct_k = target in top_candidate
                    if k == 1:
                        hit1.append(correct_k)
                        hit1_goal[goal].append(correct_k)
                    elif k == 5:
                        hit5.append(correct_k)
                        hit5_goal[goal].append(correct_k)
                    elif k == 10:
                        hit10.append(correct_k)
                        hit10_goal[goal].append(correct_k)
                    elif k == 20:
                        hit20.append(correct_k)
                        hit20_goal[goal].append(correct_k)

    # for i in range(10):
    #     print("T:%s\tP:%s" %(targets[i], pred[i]))

    # topic_eval(targets, pred)
    hit1 = np.average(hit1)
    hit5 = np.average(hit5)
    hit10 = np.average(hit10)
    hit20 = np.average(hit20)

    hit_movie_result = [np.average(hit1_goal["Movie recommendation"]), np.average(hit5_goal["Movie recommendation"]), np.average(hit10_goal["Movie recommendation"]), np.average(hit20_goal["Movie recommendation"])]
    hit_music_result = [np.average(hit1_goal["Music recommendation"]), np.average(hit5_goal["Music recommendation"]), np.average(hit10_goal["Music recommendation"]), np.average(hit20_goal["Music recommendation"])]
    hit_qa_result = [np.average(hit1_goal["Q&A"]), np.average(hit5_goal["Q&A"]), np.average(hit10_goal["Q&A"]), np.average(hit20_goal["Q&A"])]
    # hit_chat_result = [np.average(hit1_goal["Chat about stars"]), np.average(hit5_goal["Chat about stars"]), np.average(hit10_goal["Chat about stars"]), np.average(hit20_goal["Chat about stars"])]
    hit_poi_result = [np.average(hit1_goal["POI recommendation"]), np.average(hit5_goal["POI recommendation"]), np.average(hit10_goal["POI recommendation"]), np.average(hit20_goal["POI recommendation"])]

    hit_movie_result = ["%.4f" % hit for hit in hit_movie_result]
    hit_music_result = ["%.4f" % hit for hit in hit_music_result]
    hit_qa_result = ["%.4f" % hit for hit in hit_qa_result]
    # hit_chat_result = ["%.4f" % hit for hit in hit_chat_result]
    hit_poi_result = ["%.4f" % hit for hit in hit_poi_result]

    if write:
        # TODO HJ: 입출력 저장 args처리 필요시 args.save_know_output 에 store_true 옵션으로 만들 필요
        write_pkl(obj=jsonlineSave, filename='jsonline.pkl')  # 입출력 저장
        save_json(args, f"{args.time}_{args.model_name}_inout", jsonlineSave)
    else:
        print(f"Test Hit@1: %.4f" % np.average(hit1))
        print(f"Test Hit@5: %.4f" % np.average(hit5))
        print(f"Test Hit@10: %.4f" % np.average(hit10))
        print(f"Test Hit@20: %.4f" % np.average(hit20))

        print("Movie recommendation\t" + "\t".join(hit_movie_result))
        print("Music recommendation\t" + "\t".join(hit_music_result))
        print("Q&A\t" + "\t".join(hit_qa_result))
        # print("Chat about stars\t" + "\t".join(hit_chat_result))
        print("POI recommendation\t" + "\t".join(hit_poi_result))

    return [hit1, hit5, hit10, hit20, hit_movie_result, hit_music_result, hit_qa_result, hit_poi_result]
