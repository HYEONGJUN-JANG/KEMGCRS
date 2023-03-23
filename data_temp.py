from tqdm import tqdm
from torch.utils.data import DataLoader
import json
from data_util import readDic
from utils import *
import random
import torch
from torch.utils.data import Dataset
from collections import defaultdict

class DialogDataset_TEMP(Dataset): # knowledge용 데이터셋
    # << Conversation Sample Keys >>
    # 'dialog': conversation,
    # 'role_seq': role_seq,
    # 'type': dialog['goal_type_list'],
    # 'topic': dialog['goal_topic_list'],
    # 'situation': situation,
    # 'user_profile': user_profile,
    # 'knowledge_seq': knowledge_seq
    def __init__(self, args, conversation_sample, knowledgeDB, tokenizer, task, mode):
        super(Dataset, self).__init__()
        self.args = args
        self.raw_data= conversation_sample
        self.tokenizer = tokenizer
        self.knowledgeDB = knowledgeDB
        self.task = task
        self.mode = mode
        self.augmented_raw_sample = self.augment_raw_dataset(conversation_sample)
        # self.toked_sample = self.tokenize_train_sample(self.augmented_raw_sample)

    # def tokenize_train_sample(self, augmented_raw_sample): # Should be use with collate_fn
    #     output = list()
    #     for conv in augmented_raw_sample:
    #         samples = defaultdict(list)
    #         dialog, user_profile, response, type, topic, situation, target_knowledge = [conv[i] for i in ['dialog', 'user_profile', 'response', 'type', 'topic', 'situation', 'target_knowledge']]
    #         for k in ['dialog', 'user_profile', 'response', 'type', 'topic', 'situation'] : #, 'target_knowledge'
    #             samples[k].extend(self.tokenizer(conv[k], add_special_tokens=False).input_ids)
    #         samples['target_knowledge'].append(conv['target_knowledge'])
    #         output.append(samples)
    #     return output
    def augment_raw_dataset(self, raw_data):
        # checkPath(os.path.join(self.args.data_dir,'cache'))
        # cachename = os.path.join(self.args.data_dir, 'cache', f"cached_{self.task}_{self.mode}.pkl")
        # if self.args.data_cache and os.path.exists(cachename):
        #     print(f"Read Pickle {cachename}")
        #     train_sample = read_pkl(cachename)
        # else:
        train_sample = []
        for ij in range(len(raw_data)):
            conversation = raw_data[ij]
            augmented_dialog = []
            for i in range(len(conversation['dialog'])):

                role = conversation['role_seq'][i]
                if role == 'System' and len(augmented_dialog) > 0:
                    flatten_dialog = self.tokenizer.sep_token.join(augmented_dialog)
                    if self.task=='know' and conversation['knowledge_seq'][i] != '':
                        train_sample.append({'dialog': flatten_dialog,
                                                 'user_profile': conversation['user_profile'],
                                                 'response': conversation['dialog'][i],
                                                 'type': conversation['type'][i],
                                                 'topic': conversation['topic'][i],
                                                 'situation': conversation['situation'],
                                                 'target_knowledge': self.knowledgeDB.index(conversation['knowledge_seq'][i])})

                    if self.task in ['type','topic']: # goal, topic
                        train_sample.append({'dialog': flatten_dialog,
                                                 'user_profile': conversation['user_profile'],
                                                 'response': conversation['dialog'][i],
                                                 'type': conversation['type'][i],
                                                 'topic': conversation['topic'][i],
                                                 'situation': conversation['situation'],
                                                 'target_knowledge': self.knowledgeDB.index(conversation['knowledge_seq'][i])})
                augmented_dialog.append(conversation['dialog'][i])
        return train_sample
        # dialog, user_profile, response, type, topic, situation, target_knowledge



    def __getitem__(self, idx): # TODO 구현 전
        return self.augmented_raw_sample[idx]
        # return self.toked_sample[idx]

    def __len__(self):
        return len(self.augmented_raw_sample)

    def negative_sampler(self, target_knowledge):
        # candidate_entity = self.knowledgeDB[target_knowledge][0]
        # candiate_all_list = self.knowledgeDB_entity_values[candidate_entity]
        # negative_indice = random.choices(candiate_all_list, k=self.args.negative_num if len(candiate_all_list) > self.args.negative_num else len(candiate_all_list))
        total_knowledge_num = self.args.knowledge_num
        negative_indice = []
        while len(negative_indice) < self.args.negative_num:
            negative_idx = random.randint(0, total_knowledge_num-1)
            if (negative_idx not in negative_indice) and (negative_idx != target_knowledge):
                negative_indice.append(negative_idx)
        return negative_indice

    def __len__(self):
        return len(self.augmented_raw_sample)

def truncationPadding(input_ids, max_length, prefix=[], suffix=[]):
    truncate_size = max_length - len(prefix) - len(suffix)
    if truncate_size <= len(input_ids): input_ids = prefix + input_ids[len(input_ids) - truncate_size : ] + suffix
    else: input_ids = prefix + input_ids + suffix
    return input_ids + [0] * (max_length - len(input_ids))

def dataset_reader_raw_temp(args, tokenizer, knowledgeDB, data_name='train'):
    conversation_sample = []
    data_path = os.path.join(args.data_dir, f"en_{data_name}.txt")
    with open(data_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f, desc="Dataset Read", bar_format='{l_bar} | {bar:23} {r_bar}'):
            dialog = json.loads(line)
            conversation = dialog['conversation']

            role_seq = ["User", "System"] if dialog['goal_type_list'][0] != 'Greetings' else ["System", "User"]

            for i in range(2, len(conversation)):
                role_seq.append(role_seq[i % 2])

            knowledge_seq = dialog['knowledge']
            knowledge_seq = [' '.join(know) for know in knowledge_seq]
            user_profile = user_profile_setting(dialog['user_profile'])
            situation = dialog['situation']

            for i in range(len(conversation)): # HJ: [1],[2] 같은 text 제거, conversation 추가해넣는코드
                conversation[i] = conversation[i] if conversation[i][0] != '[' else conversation[i][4:]
                conversation[i] = role_seq[i] + ": " + conversation[i]
            conversation_sample.append({
                'dialog': conversation,
                'role_seq': role_seq,
                'type': dialog['goal_type_list'],
                'topic': dialog['goal_topic_list'],
                'situation': situation,
                'user_profile': user_profile,
                'knowledge_seq': knowledge_seq
            })
    return conversation_sample

def user_profile_setting(ufDic:dict)->str:
    uf=''
    for i,key in enumerate(ufDic.keys()):
        one=ufDic[key]
        if i==0 or key[0].lower()!="a": pass
        else: uf+=' | '
        if type(one)==list: uf += f"{key}: {', '.join(one[:-5])}"
        else: uf += f"{key}: {one}"
    return uf