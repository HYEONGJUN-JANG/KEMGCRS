import torch
from torch import nn
from tqdm import tqdm
import os
import torch.nn.functional as F
from utils import write_pkl, save_json
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
import logging
logger = logging.getLogger(__name__)

def train_goal(args, train_dataloader, test_dataloader, retriever, goalDic_int, tokenizer):
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.Adam(retriever.parameters(), lr=args.lr)
    jsonlineSave = []
    TotalLoss = 0
    save_output_mode = False # True일 경우 해당 epoch에서의 batch들 모아서 output으로 save
    for epoch in range(args.num_epochs):
        torch.cuda.empty_cache()
        cnt = 0
        epoch_loss = 0
        if epoch>=4: save_output_mode=True
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


        # TEST
        test_label = []
        test_pred = []
        test_loss = 0
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
                test_pred.extend(list(map(int, dot_score.argmax(1))))
                test_label.extend(list(map(int, batch['goal_type'])))
                if save_output_mode:
                    input_text = tokenizer.batch_decode(dialog_token, skip_special_tokens=True)
                    target_goal_text = [goalDic_int[idx] for idx in test_label]  # target goal
                    pred_goal_text = [goalDic_int[idx] for idx in test_pred]
                    correct = [p==l for p,l in zip(test_pred, test_label)]
                    for i in range(batch_size):
                        jsonlineSave.append(
                            {'input':input_text[i], 'pred_goal': pred_goal_text[i], 'target_goal':target_goal_text[i], 'correct':correct[i]}
                        )

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
    write_pkl(obj=jsonlineSave, filename=os.path.join(args.data_dir,'print','jsonline.pkl'))  # 입출력 저장
    save_json_hj(args, f"{args.time}_inout", jsonlineSave, "goal")
    print('done')

def train_topic(args, train_dataloader, test_dataloader, retriever, goalDic_int, topicDic_int, tokenizer):
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.Adam(retriever.parameters(), lr=args.lr)
    jsonlineSave = []
    TotalLoss = 0
    save_output_mode = False # True일 경우 해당 epoch에서의 batch들 모아서 output으로 save
    for epoch in range(args.num_epochs):
        logger.info("train epoch: {}".format(epoch))
        torch.cuda.empty_cache()
        cnt = 0
        epoch_loss = 0
        if epoch>=args.num_epochs-1: save_output_mode=True
        # TRAIN
        print("Train")
        #### return {'dialog_token': dialog_token, 'dialog_mask': dialog_mask, 'target_knowledge': target_knowledge, 'goal_type': goal_type, 'response': response, 'topic': topic, 'user_profile':user_profile, 'situation':situation}
        if args.num_epochs>1:
            retriever.train()
            for batch in tqdm(train_dataloader, desc="Topic_Train", bar_format=' {l_bar} | {bar:23} {r_bar}'):
                batch_size = batch['dialog_token'].size(0)
                dialog_token = batch['dialog_token'].to(args.device)
                dialog_mask = batch['dialog_mask'].to(args.device)
                # target_goal_type = batch['goal_type']  #
                # response = batch['response']
                target_topic = batch['topic']
                # user_profile = batch['user_profile']
                # situation = batch['situation']

                targets = torch.LongTensor(target_topic).to(args.device)
                dot_score = retriever.topic_selection(dialog_token, dialog_mask)
                loss = criterion(dot_score, targets)
                epoch_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss.detach()


        # TEST
        test_labels = []
        test_preds = []
        test_pred_at5s=[]
        test_pred_at5_tfs=[]
        test_loss = 0
        print("TEST")
        # torch.cuda.empty_cache()
        retriever.eval()
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Topic_Test", bar_format=' {l_bar} | {bar:23} {r_bar}'):
                batch_size = batch['dialog_token'].size(0)
                dialog_token = batch['dialog_token'].to(args.device)
                dialog_mask = batch['dialog_mask'].to(args.device)
                response = batch['response']
                goal_type = [goalDic_int[int(i)] for i in batch['goal_type']]
                target_topic = batch['topic']

                targets = torch.LongTensor(target_topic).to(args.device)
                test_label = list(map(int,target_topic))
                test_labels.extend(test_label)
                # user_profile = batch['user_profile']

                dot_score = retriever.topic_selection(dialog_token, dialog_mask)
                loss = criterion(dot_score, targets)
                test_loss += loss
                # test_preds.extend(list(map(int, dot_score.argmax(1))))
                test_pred= [int(i) for i in torch.topk(dot_score, k=1, dim=1).indices]
                test_pred_at5 = [list(map(int, i)) for i in torch.topk(dot_score, k=5, dim=1).indices]
                test_preds.extend(test_pred)
                test_pred_at5s.extend(test_pred_at5)
                correct = [p==l for p,l in zip(test_pred, test_label)]
                correct_at5 = [l in p for p,l in zip(test_pred_at5, test_label)]
                test_pred_at5_tfs.extend(correct_at5)
                if save_output_mode:
                    input_text = tokenizer.batch_decode(dialog_token)
                    target_topic_text = [topicDic_int[idx] for idx in test_labels]  # target goal
                    pred_topic_text = [topicDic_int[idx] for idx in test_preds]
                    pred_top5_texts = [[topicDic_int[idx] for idx in top5_idxs] for top5_idxs in test_pred_at5]
                    real_resp = tokenizer.batch_decode(response, skip_special_tokens=True)
                    for i in range(batch_size):
                        jsonlineSave.append(
                            {'input':input_text[i], 'pred': pred_topic_text[i],'pred5': pred_top5_texts[i], 'target':target_topic_text[i], 'correct':correct[i], 'response': real_resp[i], 'goal_type': goal_type[i]}
                        )
            p,r,f = round(precision_score(test_labels, test_preds, average='macro', zero_division=0), 3), round(recall_score(test_labels, test_preds, average='macro', zero_division=0), 3), round(f1_score(test_labels, test_preds, average='macro', zero_division=0), 3)
            test_hit5 = round(test_pred_at5_tfs.count(True)/len(test_pred_at5_tfs),3)
            print(f"Epoch: {epoch}\nTrain Loss: {epoch_loss}")
            print(f"Test Loss: {test_loss}")
            print(f"Test P/R/F1: {p} / {r} / {f}")
            print(f"Test Hit@5: {test_hit5}")
            logger.info("Epoch: {}, Training Loss: {}, Test Loss: {}".format(epoch, epoch_loss, test_loss))
            logger.info("Test P/R/F1:\t {} / {} / {}".format(epoch, epoch_loss, test_loss))
            logger.info("Test Hit@5: {}".format(test_hit5))
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
    write_pkl(obj=jsonlineSave, filename=os.path.join(args.data_dir,'print','jsonline.pkl'))  # 입출력 저장
    save_json_hj(args, f"{args.time}_inout", jsonlineSave, 'topic')
    print('done')


def convertDic2Tensor(dic):
    pass

def json2txt_goal(saved_jsonlines: list) -> list:
    txtlines = []
    for js in saved_jsonlines:  # TODO: Movie recommendation, Food recommendation, POI recommendation, Music recommendation, Q&A, Chat about stars
        # {'input':input_text[i], 'pred_goal': pred_goal_text[i], 'target_goal':target_goal_text[i], 'correct':correct[i]}
        dialog, pred_goal, target_goal, tf  = js['input'], js['pred_goal'], js['target_goal'], js['correct']
        txt = f"\n---------------------------\n[Target Goal]: {target_goal}\t[Pred Goal]: {pred_goal}\t[TF]: {tf}\n[Dialog]"
        for i in dialog.replace("user :", '|user :').replace("system :", "|system : ").split('|'):
            txt += f"{i}\n"
        txtlines.append(txt)
    return txtlines

def json2txt_topic(saved_jsonlines: list) -> list:
    txtlines = []
    for js in saved_jsonlines:  # TODO: Movie recommendation, Food recommendation, POI recommendation, Music recommendation, Q&A, Chat about stars
        # {'input':input_text[i], 'pred': pred_topic_text[i], 'target':target_topic_text[i], 'correct':correct[i], 'response': real_resp[i], 'goal_type': goal_type[i]}
        dialog, pred, pred5, target, tf, response,goal_type = js['input'], js['pred'],js['pred5'], js['target'], js['correct'], js['response'], js['goal_type']
        txt = f"\n---------------------------\n[Goal]: {goal_type}\t[Target Topic]: {target}\t[Pred Topic]: {pred}\t[TF]: {tf}\n[pred_top5]\n"
        for i in pred5:
            txt+=f"{i}\n"
        txt+='[Dialog]\n'
        for i in dialog.replace("user :", '||user :').replace("system :", "||system : ").split('||'):
            txt += f"{i}\n"
        txtlines.append(txt)
    return txtlines
def save_json_hj(args, filename, saved_jsonlines, task):
    '''
    Args:
        args: args
        filename: file name (path포함)
        saved_jsonlines: Key-value dictionary ( goal_type(str), topic(str), tf(str), dialog(str), target(str), response(str) predict5(list)
    Returns: None
    '''
    if task=='goal': txts = json2txt_goal(saved_jsonlines)
    elif task=='topic': txts = json2txt_topic(saved_jsonlines)
    else: return
    path = os.path.join(args.data_dir, 'print')
    if not os.path.exists(path): os.makedirs(path)
    file = f'{path}/{args.log_name}_{task}_{filename}.txt'
    with open(file, 'w', encoding='utf-8') as f:
        for i in range(len(txts)):
            f.write(txts[i])
# {'input':input_text[i], 'pred': pred_topic_text[i], 'target':target_topic_text[i], 'correct':correct[i], 'response': real_resp[i], 'goal_type': goal_type[i]}





if __name__ == "__main__":
    import main

    main.main()