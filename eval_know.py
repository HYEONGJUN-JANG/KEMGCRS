from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import write_pkl, save_json
import numpy as np


def knowledge_reindexing(args, knowledge_data, retriever):
    # 모든 know_index를 버트에 태움
    print('...knowledge indexing...')
    retriever.eval()
    knowledgeDataLoader = DataLoader(
        knowledge_data,
        batch_size=args.batch_size
    )
    knowledge_index = []

    for batch in tqdm(knowledgeDataLoader):
        input_ids = batch[0].to(args.device)
        attention_mask = batch[1].to(args.device)
        knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]
        knowledge_index.extend(knowledge_emb.cpu().detach())
    knowledge_index = torch.stack(knowledge_index, 0)
    return knowledge_index


def eval_know(args, test_dataloader, retriever, knowledge_data, knowledgeDB, tokenizer, knowledge_index=None, write=None):
    # Read knowledge DB
    # knowledge_index = knowledge_reindexing(args, knowledge_data, retriever)
    # knowledge_index = knowledge_index.to(args.device)
    jsonlineSave = []
    # bert_model = bert_model.to(args.device)
    if knowledge_index is None:
        knowledge_index = knowledge_reindexing(args, knowledge_data, retriever)
    knowledge_index = knowledge_index.to(args.device)

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
        # candidate_knowledge_token = batch['candidate_knowledge_token']  # [B,5,256]

        type_idx = [args.goalDic['int'][int(idx)] for idx in batch['type']]
        topic_idx = [args.topicDic['int'][int(idx)] for idx in batch['topic_idx']]

        # candidate_knowledge_mask = batch['candidate_knowledge_mask']  # [B,5,256]
        target_knowledge_idx = batch['target_knowledge']

        dot_score = retriever.compute_know_score(dialog_token, dialog_mask, knowledge_index, type_idx)

        if write:
            top_candidate = torch.topk(dot_score, k=args.know_topk, dim=1).indices  # [B, K]
            input_text = '||'.join(tokenizer.batch_decode(dialog_token, skip_special_tokens=True))
            target_knowledge_text = args.knowledgeDB[target_knowledge_idx]
            retrieved_knowledge_text = [knowledgeDB[idx].lower() for idx in top_candidate[0]]  # list
            correct = target_knowledge_idx in top_candidate

            response = '||'.join(tokenizer.batch_decode(response, skip_special_tokens=True))

            jsonlineSave.append({'goal_type': type_idx[0], 'topic': topic_idx[0], 'tf': correct, 'dialog': input_text, 'target': target_knowledge_text, 'response': response, "predict5": retrieved_knowledge_text})

        for idx, (score, target, goal) in enumerate(zip(dot_score, target_knowledge_idx, type_idx)):
            # goal = args.goalDic['int'][int(goal_idx)]
            # top_candidate = torch.topk(score, k=args.know_topk, dim=0).indices  # [K]
            # candidate_knowledge_text = [args.knowledgeDB[int(idx)] for idx in top_candidate]  # [K, .]
            # candidate_knowledge = tokenizer(candidate_knowledge_text, truncation=True, padding='max_length', max_length=args.max_length, return_tensors='pt')
            # candidate_knowledge_token = candidate_knowledge.input_ids.to(args.device)  # [K, L]
            # candidate_knowledge_mask = candidate_knowledge.attention_mask.to(args.device)  # [K, L]
            # re_rank_score = retriever.knowledge_retrieve(dialog_token[idx].unsqueeze(0), dialog_mask[idx].unsqueeze(0), candidate_knowledge_token.unsqueeze(0), candidate_knowledge_mask.unsqueeze(0)).squeeze(0)  # [K]

            if goal == 'Movie recommendation' or goal == 'POI recommendation' or goal == 'Music recommendation' or goal == 'Q&A' or goal == 'Chat about stars':
                for k in [1, 5, 10, 20]:
                    top_candidate_k = torch.topk(dot_score, k=k).indices  # [B, K]
                    correct_k = target in top_candidate_k
                    if k == 1:
                        hit1.append(correct_k)
                        hit1_goal[goal].append(correct_k)
                    if k == 5:
                        hit5.append(correct_k)
                        hit5_goal[goal].append(correct_k)
                    if k == 10:
                        hit10.append(correct_k)
                        hit10_goal[goal].append(correct_k)
                    if k == 20:
                        hit20.append(correct_k)
                        hit20_goal[goal].append(correct_k)

    # for i in range(10):
    #     print("T:%s\tP:%s" %(targets[i], pred[i]))

    # topic_eval(targets, pred)

    if write:
        # TODO HJ: 입출력 저장 args처리 필요시 args.save_know_output 에 store_true 옵션으로 만들 필요
        write_pkl(obj=jsonlineSave, filename='jsonline.pkl')  # 입출력 저장
        save_json(args, f"{args.time}_{args.model_name}_inout", jsonlineSave)
    else:
        print(f"Test Hit@1: %.4f" % np.average(hit1))
        print(f"Test Hit@5: %.4f" % np.average(hit5))
        print(f"Test Hit@10: %.4f" % np.average(hit10))
        print(f"Test Hit@20: %.4f" % np.average(hit20))
        hit1_goal_result = [(goal, np.average(hit1_goal[goal])) for goal in goal_list]
        hit5_goal_result = [(goal, np.average(hit5_goal[goal])) for goal in goal_list]
        hit10_goal_result = [(goal, np.average(hit10_goal[goal])) for goal in goal_list]
        hit20_goal_result = [(goal, np.average(hit20_goal[goal])) for goal in goal_list]

        result_by_goal = '\n'.join(["%s\t%.4f\t%.4f\t%.4f\t%.4f" % (h1r[0], h1r[1], h2r[1], h3r[1], h4r[1]) for h1r, h2r, h3r, h4r in zip(hit1_goal_result, hit5_goal_result, hit10_goal_result, hit20_goal_result)])
        print(result_by_goal)

    return [np.average(hit1), np.average(hit5), np.average(hit10), np.average(hit20), hit1_goal_result, hit5_goal_result, hit10_goal_result, hit20_goal_result]
