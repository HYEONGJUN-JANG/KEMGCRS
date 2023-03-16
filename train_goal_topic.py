import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from utils import write_pkl, save_json
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score


def train_goal(args, train_dataloader, test_dataloader, retriever, goalDic, tokenizer):
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.Adam(retriever.parameters(), lr=args.lr)
    jsonlineSave = []
    TotalLoss = 0
    for epoch in range(5):
        torch.cuda.empty_cache()
        cnt = 0
        epoch_loss = 0

        # TRAIN
        print("Train")
        #### return {'dialog_token': dialog_token, 'dialog_mask': dialog_mask, 'target_knowledge': target_knowledge, 'goal_type': goal_type, 'response': response, 'topic': topic, 'user_profile':user_profile, 'situation':situation}
        for batch in tqdm(train_dataloader, desc="Topic_Train", bar_format=' {l_bar} | {bar:23} {r_bar}'):
            retriever.train()
            batch_size = batch['dialog_token'].size(0)
            dialog_token = batch['dialog_token'].to(args.device)
            dialog_mask = batch['dialog_mask'].to(args.device)
            target_goal_type = batch['goal_type']  #
            # response = batch['response']
            # target_topic = batch['topic']
            # user_profile = batch['user_profile']
            # situation = batch['situation']
            targets = torch.LongTensor(target_goal_type).to(args.device)

            dot_score = retriever.goal_selection(dialog_token, dialog_mask)
            loss = criterion(dot_score, targets)
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss.detach()
        test_label = []
        test_pred = []
        test_loss = 0

        # TEST
        print("TEST")
        # torch.cuda.empty_cache()
        retriever.eval()
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Topic_Test", bar_format=' {l_bar} | {bar:23} {r_bar}'):
                batch_size = batch['dialog_token'].size(0)
                dialog_token = batch['dialog_token'].to(args.device)
                dialog_mask = batch['dialog_mask'].to(args.device)
                targets = torch.LongTensor(batch['goal_type']).to(args.device)

                dot_score = retriever.goal_selection(dialog_token, dialog_mask)
                loss = criterion(dot_score, targets)
                test_loss += loss
                test_pred.append(int(dot_score.argmax(1)[0]))
                test_label.append(int(batch['goal_type'][0]))

        print(f"Epoch: {epoch}\nTrain Loss: {epoch_loss}")
        print(f"Test Loss: {test_loss}")
        print(f"P/R/F1: {round(precision_score(test_label, test_pred, average='macro'), 3)} / {round(recall_score(test_label, test_pred, average='micro'), 3)} / {round(f1_score(test_label, test_pred, average='micro'), 3)}")
        TotalLoss += epoch_loss / len(train_dataloader)

        # pred_goal = dot_score.argmax(1) # dtype=torch.int64
        # torch.tensor([goalDic[i] for i in target_goal_type])
        # dot_score = retriever.knowledge_retrieve(dialog_token, dialog_mask, knowledge_index) # topic_selection(self, token_seq, mask, topic_idx)
        # top_candidate = torch.topk(dot_score, k=args.know_topk, dim=1).indices  # [B, K]

        # input_text = '||'.join(tokenizer.batch_decode(dialog_token, skip_special_tokens=True))
        # target_knowledge_text = [knowledgeDB[idx] for idx in target_topic]  # target knowledge
        # retrieved_knowledge_text = [knowledgeDB[idx] for idx in top_candidate[0]]  # list
        # correct = target_knowledge_text[0] in retrieved_knowledge_text
        #
        # jsonlineSave.append({'goal_type': goal_type[0], 'topic': topic, 'tf': correct, 'dialog': input_text, 'target': '||'.join(target_knowledge_text), 'response': response[0], "predict5": retrieved_knowledge_text})
        # cnt += 1

    # TODO HJ: 입출력 저장 args처리 필요시 args.save_know_output 에 store_true 옵션으로 만들 필요
    # write_pkl(obj=jsonlineSave, filename='jsonline.pkl')  # 입출력 저장
    # save_json(args, f"{args.time}_inout", jsonlineSave)
    # print('done')


def train_topic():
    pass


def convertDic2Tensor(dic):
    pass


if __name__ == "__main__":
    import main

    main.main()