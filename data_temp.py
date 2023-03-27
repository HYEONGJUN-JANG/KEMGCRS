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


def convert_to_features_unimind(args, tokenizer, mode):
    cls_ids = tokenizer.encode("<s>")[1:-1]
    sep_ids = tokenizer.encode("</s>")[1:-1]

    topic_dict = readDic(os.path.join(args.data_dir, 'topic2id.txt'))
    goal_dict = readDic(os.path.join(args.data_dir, 'goal2id.txt'))

    # if args.data_name == 'durecdial':
    #     path = os.path.join(args.data_dir, 'kb_{}.jsonl'.format(args.data_name))
    #     outfile = open(path, 'w', encoding='utf-8')
    path = os.path.join(args.data_dir, 'forunimind', '{}_hj.jsonl'.format(mode))
    print('tokenizing {}'.format(path))
    # print(tokenizer.SPECIAL_TOKENS_ATTRIBUTES)
    data_dict = {'resp': {'source_ids': [], 'target_ids': [], 'item_ids': []}, 'item': {'source_ids': [], 'target_ids': [], 'item_ids': []}, 'goal': {'source_ids': [], 'target_ids': [], 'item_ids': []}, 'know': {'source_ids': [], 'target_ids': [], 'item_ids': []}}

    # if args.tmpdata: path = os.path.join(args.data_dir, '{}/tmp{}_hj.jsonl'.format(args.data_name, mode))  # durecdial/tmptest_hj.jsonl
    # else: path = os.path.join(args.data_dir, '{}/{}_hj.jsonl'.format(args.data_name, mode))  # durecdial/test.jsonl

    with open(path, 'r', encoding='utf-8') as infile:
        max_dia_len = 0
        avg_dia_len = []
        max_res_len = 0
        avg_res_len = []
        source_ids = []
        target_ids = []
        item_ids = []
        hist_ids = []
        rec_index = []
        i = 0
        for line in tqdm(infile, desc="convert_to_features", bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            d = eval(line.strip())
            know = d['knowledge']
            conv = d['conversation']
            source_id = []
            source_know_id = []
            source_goal_id = []
            target_id = []
            hist_id = 0
            # hist_id = tokenizer.encode('[history]' + '|'.join(['<'+str(x)+'>' for x in know['item_history']]))[1:]
            profile_id = tokenizer.encode('[profile]' + '|'.join(know['user_profile']))[1:]

            first_utt = conv[0]
            if first_utt['role'] == 'user': pass
            else:
                if type(first_utt['goal']) is list:
                    first_utt['goal'] = '|'.join(first_utt['goal'])
                source_goal_id += tokenizer.encode('[goal]' + first_utt['goal'])[1:]
                source_know_id += tokenizer.encode('[topic]' + '|'.join(first_utt['topic']))[1:]
            source_id += tokenizer.encode('[{}]'.format(first_utt['role']) + first_utt['utterance'])[1:]
            source_goal_id += tokenizer.encode('[{}]'.format(first_utt['role']) + first_utt['utterance'])[1:]
            source_know_id += tokenizer.encode('[{}]'.format(first_utt['role']) + first_utt['utterance'])[1:]

            for utt in conv[1:]:
                if utt['role'] == 'user':  # and args.data_name == 'durecdial':
                    source_id += tokenizer.encode('[user]' + utt['utterance'])[1:]
                    source_know_id += tokenizer.encode('[user]' + utt['utterance'])[1:]
                    source_goal_id += tokenizer.encode('[user]' + utt['utterance'])[1:]
                    continue
                if type(utt['goal']) is list:
                    utt['goal'] = '|'.join(utt['goal'])

                ### prepare response generation data
                target_id = tokenizer.encode(utt['utterance'])
                know_len = int(args.max_length / 2)

                if args.usekg: new_source_id = source_id + tokenizer.encode('[goal]' + utt['goal'])[1:] + tokenizer.encode('[know_text]')[1:-1] + tokenizer.encode('|'.join(utt['know_text']))[1:][-know_len:] + tokenizer.encode('[item]' + '|'.join(utt['item']))[1:] + tokenizer.encode('create response：')[1:]
                else: new_source_id = source_id + tokenizer.encode('[goal]' + utt['goal'])[1:] + tokenizer.encode('[item]' + '|'.join(utt['item']))[1:] + tokenizer.encode('create response：')[1:]
                    # if mode == 'test':
                    #     outfile.write(str(know['knowledge']) + '\n')

                source_ids.append([tokenizer.cls_token_id] + new_source_id[-args.max_length + 1:])
                target_ids.append([tokenizer.cls_token_id] + target_id[-args.max_length//2 + 1:])
                item_ids.append([len(topic_dict['str']) - 1])
                data_dict['resp']['source_ids'].append(source_ids[-1])
                data_dict['resp']['target_ids'].append(target_ids[-1])
                data_dict['resp']['item_ids'].append(item_ids[-1])

                avg_dia_len.append(len(new_source_id))
                max_dia_len = max(max_dia_len, len(new_source_id))
                avg_res_len.append(len(target_id))
                max_res_len = max(max_res_len, len(target_id))

                ### prepare goal selection data
                target_id = tokenizer.encode(utt['goal'])
                new_source_id = source_goal_id + tokenizer.encode('predict next goal')[1:]
                source_goal_id += (tokenizer.encode('[goal]' + utt['goal'])[1:] + tokenizer.encode('[{}]'.format(utt['role']) + utt['utterance'])[1:])
                source_ids.append([tokenizer.cls_token_id] + new_source_id[-args.max_length + 1:])
                target_ids.append([tokenizer.cls_token_id] + target_id[-args.max_length//2 + 1:])
                item_ids.append([len(topic_dict['str']) - 1])
                data_dict['goal']['source_ids'].append(source_ids[-1])
                data_dict['goal']['target_ids'].append(target_ids[-1])
                data_dict['goal']['item_ids'].append(item_ids[-1])

                ### prepare topic prediction data
                target_id = tokenizer.encode('|'.join(utt['topic']))
                new_source_id = profile_id + source_know_id + tokenizer.encode('[goal]' + utt['goal'])[1:] + tokenizer.encode('predict next topic')[1:]
                # new_source_id = profile_id + source_know_id + tokenizer.encode('[knowledge]')[1:]
                source_know_id += (tokenizer.encode('[topic]' + '|'.join(utt['topic']))[1:] + tokenizer.encode('[{}]'.format(utt['role']) + utt['utterance'])[1:])
                source_ids.append([tokenizer.cls_token_id] + new_source_id[-args.max_length + 1:])
                target_ids.append([tokenizer.cls_token_id] + target_id[-args.max_length//2 + 1:])
                item_ids.append([len(topic_dict['str']) - 1])
                data_dict['know']['source_ids'].append(source_ids[-1])
                data_dict['know']['target_ids'].append(target_ids[-1])
                data_dict['know']['item_ids'].append(item_ids[-1])

                ### prepare item recommendation data
                if len(utt['item_id']) > 0:
                    target_text = []
                    for item, item_id in zip(utt['item'], utt['item_id']):
                        target_text.append('<' + str(item_id) + '>' + item)
                    target_id = tokenizer.encode('|'.join(target_text))
                    new_source_id = profile_id + source_id + tokenizer.encode('[goal]' + utt['goal'])[1:] + tokenizer.encode('[topic]' + '|'.join(utt['topic']))[1:] + tokenizer.encode('predict next item(topic)：')[1:]  #
                    item_id = utt['item_id']
                    source_ids.append([tokenizer.cls_token_id] + new_source_id[-args.max_length + 1:])
                    target_ids.append([tokenizer.cls_token_id] + target_id[-args.max_length//2 + 1:])
                    item_ids.append(item_id)
                    data_dict['item']['source_ids'].append(source_ids[-1])
                    data_dict['item']['target_ids'].append(target_ids[-1])
                    data_dict['item']['item_ids'].append(item_ids[-1])
                    rec_index.append(i)
                i += 1

                source_id += tokenizer.encode('[{}]'.format(utt['role']) + utt['utterance'])[1:]

                # hist_ids.append(hist_id)
                # hist_id.extend(item_id)

        print('{} set, max_res_len: {}, max_dia_len: {}, avg_res_len: {}, avg_dia_len: {}'.format(mode, max_res_len, max_dia_len, float(sum(avg_res_len)) / len(avg_res_len), float(sum(avg_dia_len)) / len(avg_dia_len)))

    if mode == 'train':
        # return {'source_ids':source_ids, 'target_ids':target_ids, 'item_ids':item_ids, 'item_dict':item_dict}
        data_dict['item_dict'] = topic_dict['str']
        return data_dict
    else:
        data_dict['item_dict'] = topic_dict['str']
        data_dict['rec_index'] = rec_index
        return data_dict

# if __name__ == "__main__":
#     from transformers import AutoModel, AutoTokenizer
#     from config import bert_special_tokens_dict
#
#     args = parseargs()
#     tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
#     tokenizer.add_special_tokens(bert_special_tokens_dict)
#     features_train = convert_to_features_unimind(args, tokenizer, 'train')
#     print(f"training samples: {len(features_train['resp']['source_ids'])}")
#     features_Test = convert_to_features_unimind(args, tokenizer, 'test')
#     print(f"test samples: {len(features_Test['resp']['source_ids'])}")
#     pass