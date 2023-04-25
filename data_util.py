import json
import os

import torch
from collections import defaultdict
import random

from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def readDic(filename, out=None):
    output_idx_str = dict()
    output_idx_int = dict()
    with open(filename, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                k, idx = line.strip().split('\t')
            except:
                print(line)
                k, idx = line.strip().split()
            output_idx_str[k] = int(idx)
            output_idx_int[int(idx)] = k
        # output_idx_str[len(output_idx_str)] = '<PAD>'
        # output_idx_int[len(output_idx_int)] = '<PAD>'
    if out == 'str':
        return output_idx_str
    elif out == 'idx':
        return output_idx_int
    else:
        return {'str': output_idx_str, 'int': output_idx_int}


def v2conv(conv, istopic=True, isgoal=True, iskg=True):
    """
    Args:
        conv: v2lines[i]
        istopic: text Topic 출력여부
        isgoal: text type(goal)출력여부
        iskg: text kg 출력여부
    Returns: {'uttrs':uttrs, 'roles':roles, 'topics':topics, 'goals':goals, 'kgs':kgs, 'situation':situation, 'user_profile':usr_profile}
    """
    usr_profile = conv.get('user_profile')
    situation = conv.get('situation')
    topics = conv.get('goal_topic_list') if istopic else ["" for _ in range(len(conv.get('goal_type_list')))]
    goals = conv.get('goal_type_list') if isgoal else ["" for _ in range(len(conv.get('goal_type_list')))]
    kgs = conv.get('knowledge')  # if iskg else ["" for _ in range(len(conv.get('goal_type_list')))]
    uttrs = [i if i[0] != '[' else i[4:] for i in conv.get('conversation')]  # utterance 내 [1] 과 같은 형태 삭제
    roles = ["system", 'user'] if goals[0] == 'Greetings' else ['user', 'system']
    for i in range(len(kgs) - 2): roles.append(roles[i % 2])
    return {'uttrs': uttrs, 'roles': roles, 'topics': topics, 'goals': goals, 'kgs': kgs, 'situation': situation, 'user_profile': usr_profile}


def truncationPadding(input_ids, max_length, prefix=[], suffix=[]):
    truncate_size = max_length - len(prefix) - len(suffix)
    if truncate_size <= len(input_ids):
        input_ids = prefix + input_ids[len(input_ids) - truncate_size:] + suffix
    else:
        input_ids = prefix + input_ids + suffix
    return input_ids + [0] * (max_length - len(input_ids))


def user_profile_setting(ufDic: dict) -> str:
    uf = ''
    for i, key in enumerate(ufDic.keys()):
        one = ufDic[key]
        if i == 0 or key[0].lower() != "a":
            pass
        else:
            uf += ' | '
        if type(one) == list:
            uf += f"{key}: {', '.join(one[:-5])}"
        else:
            uf += f"{key}: {one}"
    return uf


def process_augment_sample(raw_data, tokenizer, knowledgeDB):
    train_sample = []
    if tokenizer.eos_token is not None:
        eos_token = tokenizer.eos_token
    else:
        eos_token = tokenizer.sep_token
    for ij in range(len(raw_data)):
        conversation = raw_data[ij]
        augmented_dialog = []
        for i in range(len(conversation['dialog'])):
            role = conversation['role_seq'][i]
            utterance = conversation['dialog'][i] + eos_token

            if role == 'System' and len(augmented_dialog) > 0 and len(conversation['pseudo_knowledge_seq'][i]) != 0:
                flatten_dialog = ''.join(augmented_dialog)
                train_sample.append({'dialog': flatten_dialog,
                                     'user_profile': conversation['user_profile'],
                                     'response': utterance,
                                     'type': conversation['type'][i],
                                     'topic': conversation['topic'][i],
                                     'situation': conversation['situation'],
                                     'target_knowledge': knowledgeDB.index(conversation['knowledge_seq'][i]),
                                     'candidate_knowledges': [knowledgeDB.index(cand) for cand in conversation['pseudo_knowledge_seq'][i]],
                                     'candidate_confidences': conversation['pseudo_confidence_seq'][i]})
            augmented_dialog.append(utterance)
    return train_sample


def dataset_reader(args, data_name='train'):
    conversation_sample = []
    data_path = os.path.join(args.data_dir, f"en_{data_name}_know_cand_thresh.txt")
    with open(data_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f, desc="Dataset Read", bar_format='{l_bar} | {bar:23} {r_bar}'):
            dialog = json.loads(line)
            conversation = dialog['conversation']
            role_seq = ["User", "System"] if dialog['goal_type_list'][0] != 'Greetings' else ["System", "User"]

            for i in range(2, len(conversation)):
                role_seq.append(role_seq[i % 2])

            knowledge_seq = dialog['knowledge']
            know_candidates = dialog['know_candidates']
            pseudo_knowledge_seq = []
            pseudo_confidence_seq = []
            for idx, know_conf_list in enumerate(know_candidates):
                positive_candidates = [know[0] for know in know_conf_list]
                positive_candidates = [' '.join(candidate) for candidate in positive_candidates]
                conf_list = [know[1] for know in know_conf_list]
                pseudo_knowledge_seq.append(positive_candidates)
                pseudo_confidence_seq.append(conf_list)
                # if len(positive_candidates) > 0:
                #     positive_candidates_list[idx] = [' '.join(candidate) for candidate in positive_candidates]
                #     # positive_candidates_list[idx] = [args.knowledgeDB.index(candidate) for candidate in positive_candidates]

            knowledge_seq = [' '.join(know) for know in knowledge_seq]
            # pseudo_knowledge_seq = [' '.join(know) for know in pseudo_knowledge_seq]

            user_profile = user_profile_setting(dialog['user_profile'])
            situation = dialog['situation']

            for i in range(len(conversation)):  # HJ: [1],[2] 같은 text 제거, conversation 추가해넣는코드
                conversation[i] = conversation[i] if conversation[i][0] != '[' else conversation[i][4:]
                conversation[i] = role_seq[i] + ": " + conversation[i]
            conversation_sample.append({
                'dialog': conversation,
                'role_seq': role_seq,
                'type': dialog['goal_type_list'],
                'topic': dialog['goal_topic_list'],
                'situation': situation,
                'user_profile': user_profile,
                'knowledge_seq': knowledge_seq,
                'pseudo_knowledge_seq': pseudo_knowledge_seq,
                'pseudo_confidence_seq': pseudo_confidence_seq
            })

    return conversation_sample
