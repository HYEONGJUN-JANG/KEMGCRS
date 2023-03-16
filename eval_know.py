import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import write_pkl, save_json


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
        knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        knowledge_index.extend(knowledge_emb.cpu().detach())
    knowledge_index = torch.stack(knowledge_index, 0)
    return knowledge_index


def eval_know(args, test_dataloader, retriever, knowledgeDB, tokenizer):

    # Read knowledge DB
    # knowledge_index = knowledge_reindexing(args, knowledge_data, retriever)
    # knowledge_index = knowledge_index.to(args.device)
    jsonlineSave = []
    # bert_model = bert_model.to(args.device)
    # knowledge_index = knowledge_reindexing(args, knowledge_data, retriever)
    # knowledge_index = knowledge_index.to(args.device)

    cnt = 0
    for batch in tqdm(test_dataloader, desc="Knowledge_Test", bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'): # TODO: Knowledge task 분리중
        dialog_token, dialog_mask, target_knowledge, goal_type, response, topic, candidate_knowledge_token, candidate_knowledge_mask = batch
        batch_size = dialog_token.size(0)
        dialog_token = dialog_token.to(args.device)
        dialog_mask = dialog_mask.to(args.device)
        target_knowledge = target_knowledge.to(args.device)
        candidate_knowledge_token = candidate_knowledge_token.to(args.device)
        candidate_knowledge_mask = candidate_knowledge_mask.to(args.device)

        # tokenizer.batch_decode(dialog_token, skip_special_tokens=True)  # 'dialog context'
        # print([knowledgeDB[idx] for idx in target_knowledge]) # target knowledge

        dot_score = retriever.knowledge_retrieve(dialog_token, dialog_mask, candidate_knowledge_token, candidate_knowledge_mask)

        top_candidate = torch.topk(dot_score, k=args.know_topk, dim=1).indices  # [B, K]

        input_text = '||'.join(tokenizer.batch_decode(dialog_token, skip_special_tokens=True))
        target_knowledge_text = [knowledgeDB[idx] for idx in target_knowledge]  # target knowledge
        retrieved_knowledge_text = [knowledgeDB[idx] for idx in top_candidate[0]]  # list
        correct = target_knowledge_text[0] in retrieved_knowledge_text

        jsonlineSave.append({'goal_type': goal_type[0], 'topic': topic, 'tf': correct, 'dialog': input_text, 'target': '||'.join(target_knowledge_text), 'response': response[0], "predict5": retrieved_knowledge_text})
        cnt += 1

    # TODO HJ: 입출력 저장 args처리 필요시 args.save_know_output 에 store_true 옵션으로 만들 필요
    write_pkl(obj=jsonlineSave, filename='jsonline.pkl')  # 입출력 저장
    save_json(args, f"{args.time}_{args.log_name}_inout", jsonlineSave)
    print('done')
