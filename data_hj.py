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
    def __init__(self, args, train_sample, goal_dict, topic_dict):
        super(Dataset, self).__init__()
        self.train_sample = train_sample
        self.args = args
        self.goal_dic = goal_dict
        self.topic_dic = topic_dict

    def __getitem__(self, idx): # 모든 getitem되는 길이가 늘 같아야함
        data = self.train_sample[idx]
        dialog_token = torch.LongTensor(data['dialog_token'])
        dialog_mask = torch.LongTensor(data['dialog_mask'])
        # target_knowledge = data['target_knowledge']
        # negative_knowledge = torch.LongTensor(self.negative_sampler(target_knowledge))
        response = data['response']
        goal_type = data['goal_type']
        topic = data['topic']
        user_profile = data['user_profile']
        situation = data['situation']
        return {'dialog_token': dialog_token, 'dialog_mask': dialog_mask, 'goal_type': goal_type, 'topic': topic }
        # return {'dialog_token': dialog_token, 'dialog_mask': dialog_mask, 'target_knowledge': target_knowledge, 'goal_type': goal_type, 'response': response, 'topic': topic, 'user_profile':user_profile, 'situation':situation}
        # return dialog_token, dialog_mask, target_knowledge, goal_type, response, topic
        # 0: dialog_token, 1: dialog_mask, 2: target_knowledge, 3: goal_type, 4: response, 5: topic

    def negative_sampler(self, target_knowledge):
        total_knowledge_num = self.args.knowledge_num
        negative_indice = []
        while len(negative_indice) < self.args.negative_num:
            negative_idx = random.randint(0, total_knowledge_num)
            if (negative_idx not in negative_indice) and (negative_idx != target_knowledge):
                negative_indice.append(negative_idx)
        return negative_indice

    def __len__(self):
        return len(self.train_sample)

def dataset_reader_raw_hj(args, tokenizer, knowledgeDB, data_name='train', goal_dict=None, topic_dict=None, task=None):
    # if not os.path.exists(os.path.join(args.data_dir, 'cache')): os.makedirs(os.path.join(args.data_dir, 'cache'))
    checkPath(os.path.join(args.data_dir, 'cache'))
    cachename = os.path.join(args.data_dir, 'cache', f"cached_en_{data_name}_{task}.pkl")
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
            for line in tqdm(f, desc="Dataset Read", bar_format='{l_bar} {l_bar} % | {bar:23} {r_bar}'):
                dialog = json.loads(line)
                conversation = dialog['conversation']
                role_seq = ["User", "System"] if dialog['goal_type_list'][0] != 'Greetings' else ["System", "User"]
                for i in range(2, len(conversation)):
                    role_seq.append(role_seq[i % 2])
                knowledge_seq = dialog['knowledge']
                knowledge_seq = [' '.join(know) for know in knowledge_seq]
                user_profile = dialog['user_profile']
                situation = dialog['situation']
                for i in range(len(conversation)):
                    conversation[i] = conversation[i] if conversation[i][0] != '[' else conversation[i][4:]
                    conversation[i] = role_seq[i] + ": " + conversation[i]

                augmented_dialog = []
                for i in range(len(conversation)):
                    role = role_seq[i]
                    if role == 'System' and len(augmented_dialog) > 0:
                        flatten_dialog = tokenizer.sep_token.join(augmented_dialog)
                        # HJ: suffix 에 따라서 goal, topic이 말려들어가느냐 마느냐가 결정되는게 있음 --> task별로 어찌 처리해줄 수 있을지 고민필요
                        if task=="know":
                            suffix = '<type>' + dialog['goal_type_list'][i] + '<topic>' + dialog['goal_topic_list'][i]  # [TH] 일단 임시로 넣어봄
                        elif task=='goal':
                            suffix = ''
                        elif task=='topic':
                            suffix = '<type>' + dialog['goal_type_list'][i]
                        else: # Response
                            suffix= '<type>' + dialog['goal_type_list'][i] + '<topic>' + dialog['goal_topic_list'][i]
                        # Truncate and padding
                        tokenized_dialog = tokenizer(flatten_dialog, add_special_tokens=False)
                        tokenized_prefix = tokenizer(suffix, add_special_tokens=False)
                        input_ids = truncationPadding(input_ids=tokenized_dialog.input_ids, prefix=[tokenizer.cls_token_id], suffix=tokenized_prefix.input_ids, max_length=args.max_length)
                        attention_mask = truncationPadding(input_ids=tokenized_dialog.attention_mask, prefix=[1], suffix=tokenized_prefix.attention_mask, max_length=args.max_length)

                        train_sample.append({'dialog_token': input_ids,
                                             'dialog_mask': attention_mask,
                                             'response': conversation[i],
                                             'goal_type': goal_dict[dialog['goal_type_list'][i]],
                                             'topic': topic_dict[dialog['goal_topic_list'][i]],
                                             'user_profile': user_profile,
                                             'situation': situation
                                             })
                        if knowledge_seq[i] != '':
                            knowledge_sample.append({'dialog_token': input_ids,
                                                     'dialog_mask': attention_mask,
                                                     'response': conversation[i],
                                                     'goal_type': dialog['goal_type_list'][i],
                                                     'topic': dialog['goal_topic_list'][i],
                                                     'target_knowledge': knowledgeDB.index(knowledge_seq[i])})
                    augmented_dialog.append(conversation[i])

        if args.data_cache:
            write_pkl(train_sample, cachename)
            write_pkl(knowledge_sample, cachename_know)
    if args.who=='TH':
        data_sample = DialogDataset(args, knowledge_sample)
        batch_size = args.batch_size if 'train' == data_name else 1
        dataloader = DataLoader(data_sample, batch_size=batch_size)
        return dataloader
    elif args.who=='HJ':
        # data_sample = DialogDataset(args, train_sample)
        return train_sample


def truncationPadding(input_ids, max_length, prefix=[], suffix=[]):
    truncate_size = max_length - len(prefix) - len(suffix)
    input_ids = prefix + input_ids[-truncate_size:] + suffix
    input_ids = input_ids + [0] * (max_length - len(input_ids))
    return input_ids


if __name__ == "__main__":
    from transformers import AutoModel, AutoTokenizer

    args = parseargs()
    args.data_cache = False
    if not os.path.exists(os.path.join("cache", args.model_name)): os.makedirs(os.path.join("cache", args.model_name))
    bert_model = AutoModel.from_pretrained(args.model_name, cache_dir=os.path.join("cache", args.model_name))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    knowledgeDB = read_pkl("data/knowledgeDB.txt")
    topicDic, goalDic = readDic(os.path.join(args.data_dir, "topic2id.txt"), "str"), readDic(os.path.join(args.data_dir, "goal2id.txt"), "str")
    dataset_reader_raw_hj(args, tokenizer, knowledgeDB,  goal_dict=goalDic, topic_dict=topicDic)
