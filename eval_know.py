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

    hit1, hit5, hit10, hit20 = [], [], [], []
    cnt = 0

    pred = []
    targets = []

    for batch in tqdm(test_dataloader, desc="Knowledge_Test", bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):  # TODO: Knowledge task 분리중
        dialog_token = batch['input_ids']
        dialog_mask = batch['attention_mask']
        response = batch['response']
        candidate_knowledge_token = batch['candidate_knowledge_token']  # [B,5,256]
        type_idx = batch['type']
        topic_idx = batch['topic_idx']
        candidate_knowledge_mask = batch['candidate_knowledge_mask']  # [B,5,256]
        target_knowledge = candidate_knowledge_token[:, 0, :]
        target_knowledge_idx = batch['target_knowledge']

        dot_score = retriever.compute_know_score(dialog_token, dialog_mask, knowledge_index, type_idx)

        top_candidate = torch.topk(dot_score, k=args.know_topk, dim=1).indices  # [B, K]
        input_text = '||'.join(tokenizer.batch_decode(dialog_token, skip_special_tokens=True))
        target_knowledge_text = tokenizer.batch_decode(target_knowledge, skip_special_tokens=True)  # target knowledge
        retrieved_knowledge_text = [knowledgeDB[idx].lower() for idx in top_candidate[0]]  # list
        correct = target_knowledge_idx in top_candidate

        response = '||'.join(tokenizer.batch_decode(response, skip_special_tokens=True))

        type_idx = [args.goalDic['int'][int(idx)] for idx in type_idx]
        topic_idx = [args.topicDic['int'][int(idx)] for idx in topic_idx]

        jsonlineSave.append({'goal_type': type_idx[0], 'topic': topic_idx[0], 'tf': correct, 'dialog': input_text, 'target': '||'.join(target_knowledge_text), 'response': response, "predict5": retrieved_knowledge_text})
        cnt += 1

        goal = type_idx[0]
        if goal == 'Movie recommendation' or goal == 'POI recommendation' or goal == 'Music recommendation' or goal == 'Q&A' or goal == 'Chat about stars':
            for k in [1, 5, 10, 20]:
                top_candidate_k = torch.topk(dot_score, k=k, dim=1).indices  # [B, K]
                correct_k = target_knowledge_idx in top_candidate_k
                if k == 1: hit1.append(correct_k)
                if k == 5: hit5.append(correct_k)
                if k == 10: hit10.append(correct_k)
                if k == 20: hit20.append(correct_k)

    # for i in range(10):
    #     print("T:%s\tP:%s" %(targets[i], pred[i]))

    # topic_eval(targets, pred)

    print(f"Test Hit@1: %.4f" % np.average(hit1))
    print(f"Test Hit@5: %.4f" % np.average(hit5))
    print(f"Test Hit@10: %.4f" % np.average(hit10))
    print(f"Test Hit@20: %.4f" % np.average(hit20))

    if write:
        # TODO HJ: 입출력 저장 args처리 필요시 args.save_know_output 에 store_true 옵션으로 만들 필요
        write_pkl(obj=jsonlineSave, filename='jsonline.pkl')  # 입출력 저장
        save_json(args, f"{args.time}_{args.model_name}_inout", jsonlineSave)
    print('done')

    return [np.average(hit1), np.average(hit5), np.average(hit10), np.average(hit20)]
