from tqdm import tqdm
from torch.utils.data import DataLoader
# from dataModel import DialogDataset
import json

from data_util import readDic
from utils import *
import random
import torch
from torch.utils.data import Dataset

class DialogDataset(Dataset):
    def __init__(self, args, train_sample, goal_dict=None, topic_dict=None, knowledgeDB=None, tokenizer = None):
        super(Dataset, self).__init__()
        self.train_sample = train_sample
        self.args = args
        self.goal_dic = goal_dict
        self.topic_dic = topic_dict
        self.knowledgeDB = knowledgeDB
        self.tokenizer = tokenizer
        self.pred_dict = None
        self.task = args.task

    def __getitem__(self, idx): # 모든 getitem되는 길이가 늘 같아야함
        data = self.train_sample[idx]

        # HJ : data_reader역할분리 --> 순수 데이터만 관리 , DialogDataset에서 직접 tensor로 만들고 truc처리
        dialog = data['dialog']
        response = data['response']
        goal = data['goal']
        topic = data['topic']
        user_profile = data['user_profile']
        situation = data['situation']
        target_know = data['target_knowledge'] if data['target_knowledge'] else -10

        if self.args.task=='goal': suffix = ""
        elif self.args.task == 'topic': suffix = '<situation>'+situation +' <user_profile>' + user_profile + ' <type>' + goal
        elif self.args.task == 'know': suffix = self.tokenizer.sep_token + '<type>' + goal + '<topic>' + topic
        else : suffix = "" # TODO Response 시 suffix
        toked_dialog = self.tokenizer(dialog, add_special_tokens=False)
        toked_suffix = self.tokenizer(suffix, add_special_tokens=False, max_length=self.args.max_length//4)
        goal_idx = self.goal_dic[goal]
        topic_idx = self.topic_dic[topic]
        hasKnow = 1 if target_know>=0 else 0


        dialog_token = truncationPadding(input_ids=toked_dialog.input_ids, prefix=[self.tokenizer.cls_token_id],suffix=toked_suffix.input_ids, max_length=self.args.max_length)
        dialog_mask = truncationPadding(input_ids=toked_dialog.attention_mask, prefix=[1], suffix=toked_suffix.attention_mask, max_length=self.args.max_length)


        dialog_token = torch.LongTensor(dialog_token)
        dialog_mask = torch.LongTensor(dialog_mask)

        # About Knowledge 관련 처리
        # target_know = self.knowledgeDB.index(target_know) if target_know else 0
        # target_know = target_know if target_know else 0
        # negative_indice = self.negative_sampler(target_know)
        # candidate_indice = [target_know] + negative_indice
        # candidate_knowledge = self.tokenizer([self.knowledgeDB[idx] for idx in candidate_indice], truncation=True, padding='max_length', max_length=self.args.max_length)
        # candidate_knowledge_token = candidate_knowledge.input_ids
        # candidate_knowledge_mask = candidate_knowledge.attention_mask
        # candidate_knowledge_token = torch.LongTensor(candidate_knowledge_token)
        # candidate_knowledge_mask = torch.LongTensor(candidate_knowledge_mask)

        response = self.tokenizer(response, add_special_tokens=True, max_length=self.args.max_length, padding='max_length', truncation=True)
        response_token = torch.LongTensor(response.input_ids).to(self.args.device)
        response_mask = torch.LongTensor(response.attention_mask).to(self.args.device)
        return {'dialog_token': dialog_token,
                'dialog_mask': dialog_mask,
                'response_token': response_token,
                'response_mask': response_mask,
                'goal_type': goal_idx,
                'topic': topic_idx,
                # 'candidate_knowledge_token': candidate_knowledge_token,
                # 'candidate_knowledge_mask': candidate_knowledge_mask
                'hasKnow': hasKnow,
                 # ,'response': response
                 # ,'user_profile':user_profile, 'situation':situation}
                }



    def negative_sampler(self, target_knowledge):
        total_knowledge_num = self.args.knowledge_num
        negative_indice = []
        if target_knowledge:
            while len(negative_indice) < self.args.negative_num:
                negative_idx = random.randint(0, total_knowledge_num)
                if (negative_idx not in negative_indice) and (negative_idx != target_knowledge):
                    negative_indice.append(negative_idx)
            return negative_indice
        else : return [0 for i in range(self.args.negative_num)]

    def __len__(self):
        return len(self.train_sample)

def dataset_reader_raw_hj(args, tokenizer, knowledgeDB, data_name='train', goal_dict=None, topic_dict=None):
    # if not os.path.exists(os.path.join(args.data_dir, 'cache')): os.makedirs(os.path.join(args.data_dir, 'cache'))
    checkPath(os.path.join(args.data_dir, 'cache'))
    cachename = os.path.join(args.data_dir, 'cache', f"cached_en_{data_name}.pkl")
    cachename_know = os.path.join(args.data_dir, 'cache', f"cached_en_{data_name}_know.pkl")

    if args.data_cache and os.path.exists(cachename) and os.path.exists(cachename_know):
        print(f"Read Pickle {cachename}")
        train_sample = read_pkl(cachename)
        knowledge_sample = read_pkl(cachename_know)
    else:
        train_sample = []
        knowledge_sample = []
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

                augmented_dialog = []
                for i in range(len(conversation)):
                    role = role_seq[i]
                    if role == 'System' and len(augmented_dialog) > 0:
                        flatten_dialog = tokenizer.sep_token.join(augmented_dialog)

                        train_sample.append({'dialog': flatten_dialog,
                                             'user_profile': user_profile,
                                             'response': conversation[i],
                                             'goal': dialog['goal_type_list'][i],
                                             'topic': dialog['goal_topic_list'][i],
                                             'situation': situation,
                                             'target_knowledge': knowledgeDB.index(knowledge_seq[i]) if knowledge_seq[i] else None
                                             })
                        if knowledge_seq[i] != '':
                            knowledge_sample.append({'dialog': flatten_dialog,
                                             'user_profile': user_profile,
                                             'response': conversation[i],
                                             'goal_type': dialog['goal_type_list'][i],
                                             'topic': dialog['goal_topic_list'][i],
                                             'situation': situation,
                                             'target_knowledge': knowledgeDB.index(knowledge_seq[i])
                                             })
                    augmented_dialog.append(conversation[i])

        if args.data_cache:
            write_pkl(train_sample, cachename)
            write_pkl(knowledge_sample, cachename_know)
    return train_sample


def truncationPadding(input_ids, max_length, prefix=[], suffix=[]):
    truncate_size = max_length - len(prefix) - len(suffix)
    if truncate_size <= len(input_ids): input_ids = prefix + input_ids[len(input_ids) - truncate_size:] + suffix
    else: input_ids = prefix + input_ids + suffix
    return input_ids + [0] * (max_length - len(input_ids))

def user_profile_setting(ufDic:dict)->str:
    uf=''
    for i,key in enumerate(ufDic.keys()):
        one=ufDic[key]
        if i==0 or key[0].lower()!="a": pass
        else: uf+=' | '
        if type(one)==list: uf += f"{key}: {', '.join(one[:-5])}"
        else: uf += f"{key}: {one}"
    return uf


if __name__ == "__main__":
    from transformers import AutoModel, AutoTokenizer

    args = parseargs()
    args.data_cache = False
    if not os.path.exists(os.path.join("cache", args.model_name)): os.makedirs(os.path.join("cache", args.model_name))
    bert_model = AutoModel.from_pretrained(args.model_name, cache_dir=os.path.join("cache", args.model_name))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    knowledgeDB = read_pkl("data/knowledgeDB.txt")
    topicDic, goalDic = readDic(os.path.join(args.data_dir, "topic2id.txt"), "str"), readDic(os.path.join(args.data_dir, "goal2id.txt"), "str")
    dataset_reader_raw_hj(args, tokenizer, knowledgeDB,  goal_dict=goalDic, topic_dict=topicDic, task='goal')
