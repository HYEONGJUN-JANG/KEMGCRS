from datetime import datetime
from tqdm import tqdm
import json, os
import random
from collections import defaultdict
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import utils
from pytz import timezone
import random
import logging
from copy import deepcopy  # TH

logger = logging.getLogger(__name__)


def get_time_kst(): return datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d_%H%M%S')


def user_profile_setting(ufDic: dict) -> str:  # Accepted 제외하고 다 버림
    uf = '<user_profile> '
    for i, key in enumerate(ufDic.keys()):
        one = ufDic[key]
        if i == 0 or key[0].lower() != "a": continue
        if type(one) == list:
            uf += f"{key}: {', '.join(one[-5:])} |"
        else:
            uf += f"{key}: {one}|"
    return uf


def convert_know(know):
    if len(know) == 0: return ''
    if know[1] == 'Sings':
        know = ' '.join([know[0], 'singer', know[2]])
    elif know[1] == 'Stars':
        know = ' '.join([know[0], 'star', know[2]])
    elif know[1] == 'Intro':
        know = ' '.join([know[0], 'is', know[2]])
    elif know[1] == 'Comments':
        know = ' '.join([know[0], 'is known', know[2]])
    elif know[1] == 'Birthday':
        know = ' '.join([know[0], know[1], datetime.strptime(know[2].replace(' ', ''), '%Y-%m-%d').strftime('%Y %B %dth')])
    else:
        know = ' '.join(know)
    return know


def dataset_reader(args, mode='train'):
    all_knowledge = set()
    conversation_sample = []

    data_path = os.path.join('data', f"en_{mode}_know_cand_score20.txt")
    # cached_path = os.path.join(args.home, 'data', 'pseudo_du2', f"cached_en_{mode}_pseudo_len{args.max_length}.pkl")
    idx_line = 0
    logger.info(f"Read Dataset {data_path}")
    with open(data_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f, desc="Dataset Read", bar_format='{l_bar} | {bar:23} {r_bar}'):
            idx_line += 1
            dialog = json.loads(line)
            conversation = dialog['conversation']
            role_seq = ["user", "system"] if dialog['goal_type_list'][0] != 'Greetings' else ["system", "user"]

            for i in range(2, len(conversation)):
                role_seq.append(role_seq[i % 2])

            knowledge_seq = dialog['knowledge']
            know_candidates = dialog['know_candidates']
            pseudo_knowledge_seq = []
            pseudo_confidence_seq = []
            for idx, know_conf_list in enumerate(know_candidates):
                positive_candidates = [know[0] for know in know_conf_list]
                positive_candidates = [convert_know(candidate) for candidate in positive_candidates]

                conf_list = [know[1] for know in know_conf_list]
                pseudo_knowledge_seq.append(positive_candidates)
                pseudo_confidence_seq.append(conf_list)

            knowledge_seq = [convert_know(know) for know in knowledge_seq]
            all_knowledge.update(knowledge_seq)

            user_profile = user_profile_setting(dialog['user_profile'])
            situation = dialog['situation']

            ## Related Knowledge 관련부분
            related_knowledges = []  # 아래 for문으로 처리 + [convert_know(know) for know in dialog['knowledge']]

            for know in dialog['knowledge']:  # 해당 대화에 등장한 knowledge들 모아놓은것 (정답들로 이루어짐)
                if know: related_knowledges.append(convert_know(know))
            ## Related Knowledge 관련부분 끝

            for i in range(len(conversation)):  # HJ: [1],[2] 같은 text 제거, conversation 추가해넣는코드
                conversation[i] = (conversation[i] if conversation[i][0] != '[' else conversation[i][4:]).replace("\xa0", " ")
                conversation[i] = role_seq[i] + ": " + conversation[i]
            conversation_sample.append({
                'dialog': conversation,
                'role_seq': role_seq,
                'type': dialog['goal_type_list'],
                'topic': dialog['goal_topic_list'],
                'situation': situation,
                'user_profile': user_profile,
                'related_knowledges': related_knowledges,
                'knowledge_seq': knowledge_seq,
                'pseudo_knowledge_seq': pseudo_knowledge_seq,
                'pseudo_confidence_seq': pseudo_confidence_seq
            })

    return conversation_sample, all_knowledge


def process_augment_sample(raw_data, tokenizer=None):
    train_sample = []
    if tokenizer:
        try:
            if tokenizer.eos_token is not None:
                eos_token = tokenizer.eos_token
            else:
                eos_token = tokenizer.sep_token
        except:
            eos_token = tokenizer.generator.eos_token
    else:
        eos_token = '</s>'
    for ij in tqdm(range(len(raw_data)), desc="Dataset Augment", bar_format='{l_bar} | {bar:23} {r_bar}'):
        conversation = raw_data[ij]
        augmented_dialog = []
        augmented_knowledge = []
        last_type = ""
        for i in range(len(conversation['dialog'])):
            role = conversation['role_seq'][i]
            utterance = conversation['dialog'][i] + eos_token
            goal = conversation['type'][i]
            if goal == 'Movie recommendation' or goal == 'POI recommendation' or goal == 'Music recommendation' or goal == 'Q&A':  # TH 230601
                if role == 'system' and len(augmented_dialog) > 0 and len(conversation['pseudo_knowledge_seq'][i]) != 0:
                    flatten_dialog = ''.join(augmented_dialog)
                    train_sample.append({'dialog': flatten_dialog,
                                         'user_profile': conversation['user_profile'],
                                         'response': utterance,
                                         'type': conversation['type'][i],
                                         'last_type': last_type,
                                         'topic': conversation['topic'][i],
                                         'situation': conversation['situation'],
                                         'related_knowledges': conversation['related_knowledges'],
                                         'augmented_knowledges': deepcopy(augmented_knowledge),  # TH related 대신에 know seq 230601
                                         'target_knowledge': conversation['knowledge_seq'][i],
                                         'candidate_knowledges': conversation['pseudo_knowledge_seq'][i],
                                         'candidate_confidences': conversation['pseudo_confidence_seq'][i]  # prob
                                         })
            if role == 'system': last_type = conversation['type'][i]
            augmented_dialog.append(utterance)
            augmented_knowledge.append(conversation['knowledge_seq'][i])
    return train_sample


class GenerationDataset(Dataset):  # knowledge용 데이터셋
    def __init__(self, args, data_sample, train_knowledge_seq_set, tokenizer, mode='train'):
        super(Dataset, self).__init__()
        self.args = args
        self.train_knowledge_seq_set = train_knowledge_seq_set
        self.tokenizer = tokenizer
        self.augmented_raw_sample = data_sample
        self.mode = mode
        # self.generate_prompt_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('System:'))

    def __getitem__(self, idx):  # TODO 구현 전
        data = self.augmented_raw_sample[idx]
        cbdicKeys = ['dialog', 'user_profile', 'situation', 'response', 'type', 'last_type', 'topic', 'related_knowledges', 'augmented_knowledges', 'target_knowledge', 'candidate_knowledges']
        dialog, user_profile, situation, response, type, last_type, topic, related_knowledges, augmented_knowledges, target_knowledge, candidate_knowledges = [data[i] for i in cbdicKeys]
        pad_token_id = self.tokenizer.current_tokenizer.pad_token_id

        context_batch = defaultdict()

        ## Related knowledge 범위 관련 세팅##
        related_knowledge_text = ""
        if self.args.n_candidate_knowledges > 0:  ## Pseudo related knowledge 줄 때 (60%로 정답포함)
            related_knowledge_text = " | ".join(list(filter(lambda x: x, candidate_knowledges[:self.args.n_candidate_knowledges])))
        else:  ## candidate knowledge에서 사용할 때,
            related_knowledge_text = " | ".join(list(filter(lambda x: x, related_knowledges)))

        ## Related knowledge 범위 관련 세팅## -- 1. 해당대화전체knowledge, 2. 해당응답전까지knowledge 비교
        # related_knowledge를 candidate_knowledges에서 25개 무작위로 뽑아서 넣어줘서 돌리기

        related_knowledge_text += situation  # situation
        related_knowledge_text += user_profile  # user_profile

        max_knowledge_length = self.args.max_length * 5 // 10  # 768의 50%까지 knowledge데이터 넣어주기
        related_knowledge_tokens = self.tokenizer('<knowledge>' + related_knowledge_text, max_length=max_knowledge_length, truncation=True).input_ids

        type_token = self.tokenizer('<type>' + type, max_length=max_knowledge_length // 20, truncation=True).input_ids
        last_type_token = self.tokenizer('<last_type>', max_length=max_knowledge_length // 20, truncation=True).input_ids
        topic_token = self.tokenizer('<topic>' + topic, max_length=max_knowledge_length // 20, truncation=True).input_ids

        if self.args.inputWithKnowledge:
            if self.args.inputWithTopic:
                input = self.tokenizer('<dialog>' + dialog, max_length=self.args.max_length - len(related_knowledge_tokens) - len(type_token) - len(topic_token), padding='max_length', truncation=True).input_ids
                input = related_knowledge_tokens + input + type_token + topic_token  # {TH} knowledge 를 input에서 빼보는 ablation 적용을 위해 주석 (윗줄포함)
            else:
                input = self.tokenizer('<dialog>' + dialog, max_length=self.args.max_length - len(related_knowledge_tokens) - len(type_token) - len(last_type_token), padding='max_length', truncation=True).input_ids
                input = related_knowledge_tokens + input + type_token + last_type_token  # {TH} knowledge 를 input에서 빼보는 ablation 적용을 위해 주석 (윗줄포함)
        else:  # KEMGCRS세팅과 같은 input (dialog + type + topic으로 knowledge예측)
            if self.args.inputWithTopic:
                input = self.tokenizer('<dialog>' + dialog, max_length=self.args.max_length - len(type_token) - len(last_type_token) - len(topic_token), padding='max_length', truncation=True).input_ids
                input = input + type_token + last_type_token + topic_token
            else:
                input = self.tokenizer('<dialog>' + dialog, max_length=self.args.max_length - len(type_token) - len(last_type_token), padding='max_length', truncation=True).input_ids
                input = input + type_token + last_type_token

        label = self.tokenizer('<knowledge>' + target_knowledge, max_length=self.args.max_gen_length, padding='max_length', truncation=True).input_ids
        pseudo_label = self.tokenizer('<knowledge>' + candidate_knowledges[0], max_length=self.args.max_gen_length, padding='max_length', truncation=True).input_ids

        if self.args.is_rag:
            input_ids = torch.LongTensor(input)
            # response
            target_ids = torch.LongTensor(self.tokenizer(response, max_length=self.args.max_target_length, padding='max_length', truncation=True).input_ids)
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ne(input_ids, pad_token_id),
                "decoder_input_ids": target_ids,
                'knowledge_task_label': torch.LongTensor(label),
                'knowledge_task_pseudo_label': torch.LongTensor(pseudo_label),
                'is_new_knowledge': 1 if target_knowledge not in self.train_knowledge_seq_set else 0,
            }
        else:
            # <s> knowledge text~~ </s> 로 EOS가 제대로 들어가짐
            context_batch['knowledge_task_input'] = torch.LongTensor(input)
            context_batch['attention_mask'] = torch.ne(context_batch['knowledge_task_input'], pad_token_id)
            context_batch['knowledge_task_label'] = torch.LongTensor(label)
            context_batch['knowledge_task_pseudo_label'] = torch.LongTensor(pseudo_label)
            context_batch['is_new_knowledge'] = 1 if target_knowledge not in self.train_knowledge_seq_set else 0
            return context_batch

    def __len__(self):
        return len(self.augmented_raw_sample)


if __name__ == "__main__":
    args = utils.parseargs()
    # python main.py --version=2 --gpu=3 --batch_size=24 --num_beams=5 --pseudo --usePLM --inputWithKnowledge --n_candidate_knowledges=20 --log_name="CandidateKnowledge_20_PsdLabel_Du2PLM"
    if args.debug:
        args.pseudo = True
        args.usePseudoLabel = True
        args.inputWithKnowledge = True
        args.n_candidate_knowledges = 25
        args.related_sample = True
        args.usePLM = True
        args.num_beams = 5
        args.num_epochs = 1

    from transformers import BartTokenizer, BartForConditionalGeneration

    args.model = 'facebook/bart-base'
    tokenizer = BartTokenizer.from_pretrained(args.model, cache_dir=args.model_cache_dir)

    train_dataset_raw, train_knowledge_seq_set = dataset_reader(args, 'train')
    test_dataset_raw, test_knowledge_seq_set = dataset_reader(args, 'test')
    valid_dataset_raw, valid_knowledge_base = dataset_reader(args, 'dev')
    test_knowledge_seq_set = train_knowledge_seq_set.union(valid_knowledge_base).union(test_knowledge_seq_set)

    train_dataset_resp = process_augment_sample(train_dataset_raw, tokenizer)
    test_dataset_resp = process_augment_sample(test_dataset_raw, tokenizer)

    train_datamodel_resp = GenerationDataset(args, train_dataset_resp, train_knowledge_seq_set, tokenizer, mode='train')
    test_datamodel_resp = GenerationDataset(args, test_dataset_resp, train_knowledge_seq_set, tokenizer, mode='test')

    train_dataloader_resp = DataLoader(train_datamodel_resp, batch_size=args.batch_size, shuffle=True)
    test_dataloader_resp = DataLoader(test_datamodel_resp, batch_size=args.batch_size, shuffle=False)

    model = BartForConditionalGeneration.from_pretrained(args.model, cache_dir=args.model_cache_dir).to(args.device)
    test_loss = 0
    context_words, pred_words, label_words = [], [], []
    # model.to(args.device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(train_dataloader_resp, desc=f"Epoch ", bar_format=' {l_bar} | {bar:23} {r_bar}'):
            dialog = torch.as_tensor(batch['knowledge_task_input'], device=args.device)
            knowledge = torch.as_tensor(batch['knowledge_task_label'], device=args.device)
            dialog_mask = torch.as_tensor(batch['attention_mask'], device=args.device)

            outputs = model(input_ids=dialog, labels=knowledge, output_hidden_states=True)
            loss = outputs.loss
            test_loss += loss.item()

            context_word = tokenizer.batch_decode(dialog, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            label_word = tokenizer.batch_decode(knowledge, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            summary_ids = model.generate(input_ids=dialog, attention_mask=dialog_mask, num_beams=1, pad_token_id=tokenizer.pad_token_id, max_length=args.max_gen_length)
            # summary_ids = model.generate(dialog, num_beams=1, min_length=0, max_length=args.max_gen_length)
            pred_word = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            context_words.extend(context_word)
            pred_words.extend(pred_word)
            label_words.extend(label_word)