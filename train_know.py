import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim
from data_temp import DialogDataset_TEMP
from eval_know import knowledge_reindexing, eval_know
from metric import EarlyStopping
from utils import *
from models import *
import logging
import numpy as np

logger = logging.getLogger(__name__)


def update_key_bert(key_bert, query_bert):
    print('update moving average')
    decay = 0  # If 0 then change whole parameter
    for current_params, ma_params in zip(query_bert.parameters(), key_bert.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = decay * old_weight + (1 - decay) * up_weight


def train_know(args, train_dataloader, test_dataloader, retriever, knowledge_data, knowledgeDB, all_knowledge_data, all_knowledgeDB, tokenizer):
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=0)
    optimizer = optim.AdamW(retriever.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

    # eval_know(args, test_dataloader, retriever, knowledge_data, knowledgeDB, tokenizer)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리

    best_hit = [[], [], [], [], []]
    best_hit_new = [[], [], [], [], []]

    best_hit_movie = [[], [], [], [], []]
    best_hit_poi = [[], [], [], [], []]
    best_hit_music = [[], [], [], [], []]
    best_hit_qa = [[], [], [], [], []]
    best_hit_chat = [[], [], [], [], []]
    best_hit_food = [[], [], [], [], []]

    eval_metric = [-1]
    result_path = f"{args.time}_{args.model_name}_result"
    with open(os.path.join('results', result_path), 'a', encoding='utf-8') as result_f:
        result_f.write(
            '\n=================================================\n')
        result_f.write(get_time_kst())
        result_f.write('\n')
        result_f.write('Argument List:' + str(sys.argv) + '\n')
        # for i, v in vars(args).items():
        #     result_f.write(f'{i}:{v} || ')
        result_f.write('\n')
        result_f.write('[EPOCH]\tHit@1\tHit@5\tHit@10\tHit@20\n')

    for epoch in range(args.num_epochs):
        train_epoch_loss = 0
        num_update = 0
        for batch in tqdm(train_dataloader, desc="Knowledge_Train", bar_format=' {l_bar} | {bar:23} {r_bar}'):
            retriever.train()
            dialog_token = batch['input_ids']
            dialog_mask = batch['attention_mask']
            goal_idx = batch['goal_idx']
            candidate_indice = batch['candidate_indice']
            candidate_knowledge_token = batch['candidate_knowledge_token']  # [B,2,256]
            candidate_knowledge_mask = batch['candidate_knowledge_mask']  # [B,2,256]
            target_knowledge_idx = batch['target_knowledge']  # [B,5,256]

            logit_pos, logit_neg = retriever.knowledge_retrieve(dialog_token, dialog_mask, candidate_indice, candidate_knowledge_token, candidate_knowledge_mask)  # [B, 2]
            # cumsum_logit = torch.cumsum(logit_pos, dim=1)  # [B, K]  # Grouping

            loss = 0
            pseudo_confidences_mask = batch['pseudo_confidences']  # [B, K]
            for idx in range(args.pseudo_pos_rank):
                g_logit = torch.sum(logit_pos[:, :idx + 1] * pseudo_confidences_mask[:, :idx + 1], dim=-1) / (torch.sum(pseudo_confidences_mask[:, :idx + 1], dim=-1) + 1e-20)
                # g_logit = cumsum_logit[:, idx] / (idx + 1)
                # g_logit = cumsum_logit[:, idx] / batch_denominator[:, idx]
                # g_logit = cumsum_logit[:, idx] / num_samples[:, idx]
                # g_logit = cumsum_logit[:, idx]  # For Sampling
                g_logit = torch.cat([g_logit.unsqueeze(1), logit_neg], dim=1)
                loss += (-torch.log_softmax(g_logit, dim=1).select(dim=1, index=0)).mean()

            train_epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_update += 1

        scheduler.step()

        knowledge_index = knowledge_reindexing(args, knowledge_data, retriever, args.stage)
        knowledge_index = knowledge_index.to(args.device)
        # retriever.init_know_proj(knowledge_index)

        print(f"Epoch: {epoch}\nTrain Loss: {train_epoch_loss}")

        hit1, hit3, hit5, hit10, hit20, hit_movie_result, hit_music_result, hit_qa_result, hit_poi_result, hit_food_result, hit_chat_result, hit1_new, hit3_new, hit5_new, hit10_new, hit20_new = eval_know(args, test_dataloader, retriever, all_knowledge_data, all_knowledgeDB,
                                                                                                                                                                                                            tokenizer)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리

        with open(os.path.join('results', result_path), 'a', encoding='utf-8') as f:
            f.write("EPOCH:\t%d\n" % epoch)
            f.write("Overall\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (hit1, hit3, hit5, hit10, hit20))
            f.write("Overall new knowledge\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (hit1_new, hit3_new, hit5_new, hit10_new, hit20_new))

            f.write("Movie recommendation\t" + "\t".join(hit_movie_result) + "\n")
            f.write("Music recommendation\t" + "\t".join(hit_music_result) + "\n")
            f.write("Q&A\t" + "\t".join(hit_qa_result) + "\n")
            f.write("POI recommendation\t" + "\t".join(hit_poi_result) + "\n")
            f.write("Food recommendation\t" + "\t".join(hit_food_result) + "\n")
            f.write("Chat about stars\t" + "\t".join(hit_chat_result) + "\n\n")

        if hit1 > eval_metric[0]:
            eval_metric[0] = hit1
            best_hit[0] = hit1
            best_hit[1] = hit3
            best_hit[2] = hit5
            best_hit[3] = hit10
            best_hit[4] = hit20
            best_hit_new[0] = hit1_new
            best_hit_new[1] = hit3_new
            best_hit_new[2] = hit5_new
            best_hit_new[3] = hit10_new
            best_hit_new[4] = hit20_new
            best_hit_movie = hit_movie_result
            best_hit_poi = hit_poi_result
            best_hit_music = hit_music_result
            best_hit_qa = hit_qa_result
            best_hit_chat = hit_chat_result
            best_hit_food = hit_food_result

            if args.stage == 'retrieve':
                torch.save(retriever.state_dict(), os.path.join(args.model_dir, f"{args.model_name}_retriever.pt"))  # TIME_MODELNAME 형식

            # best_hit_chat = hit_chat_result

    print(f'BEST RESULT')
    print(f"BEST Test Hit@1: {best_hit[0]}")
    print(f"BEST Test Hit@3: {best_hit[1]}")
    print(f"BEST Test Hit@5: {best_hit[2]}")
    print(f"BEST Test Hit@10: {best_hit[3]}")
    print(f"BEST Test Hit@20: {best_hit[4]}")

    checkPath('results')
    with open(os.path.join('results', result_path), 'a', encoding='utf-8') as f:
        f.write("[BEST]\n")
        f.write("Overall\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (best_hit[0], best_hit[1], best_hit[2], best_hit[3], best_hit[4]))
        f.write("New\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (best_hit_new[0], best_hit_new[1], best_hit_new[2], best_hit_new[3], best_hit_new[4]))

        f.write("Movie recommendation\t" + "\t".join(best_hit_movie) + "\n")
        f.write("Music recommendation\t" + "\t".join(best_hit_music) + "\n")
        f.write("QA\t" + "\t".join(best_hit_qa) + "\n")
        f.write("POI recommendation\t" + "\t".join(best_hit_poi) + "\n")
        f.write("Food recommendation\t" + "\t".join(best_hit_food) + "\n")
        f.write("Chat about stars\t" + "\t".join(best_hit_chat) + "\n")
