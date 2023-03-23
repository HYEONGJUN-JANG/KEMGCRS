import torch
from collections import defaultdict
import random

def readDic(filename, out=None):
    output_idx_str=dict()
    output_idx_int=dict()
    with open(filename, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                k, idx=line.strip().split('\t')
            except:
                print(line)
                k, idx=line.strip().split()
            output_idx_str[k]=int(idx)
            output_idx_int[int(idx)]=k
        # output_idx_str[len(output_idx_str)] = '<PAD>'
        # output_idx_int[len(output_idx_int)] = '<PAD>'
    if out=='str':
        return output_idx_str
    elif out=='idx':
        return output_idx_int
    else:
        return {'str':output_idx_str, 'int':output_idx_int}

def v2conv(conv, istopic=True, isgoal=True, iskg=True):
    """
    Args:
        conv: v2lines[i]
        istopic: text Topic 출력여부
        isgoal: text type(goal)출력여부
        iskg: text kg 출력여부
    Returns: {'uttrs':uttrs, 'roles':roles, 'topics':topics, 'goals':goals, 'kgs':kgs, 'situation':situation, 'user_profile':usr_profile}
    """
    usr_profile=conv.get('user_profile')
    situation=conv.get('situation')
    topics = conv.get('goal_topic_list') if istopic else ["" for _ in range(len(conv.get('goal_type_list')))]
    goals = conv.get('goal_type_list') if isgoal else ["" for _ in range(len(conv.get('goal_type_list')))]
    kgs = conv.get('knowledge')# if iskg else ["" for _ in range(len(conv.get('goal_type_list')))]
    uttrs = [i if i[0] != '[' else i[4:] for i in conv.get('conversation') ] # utterance 내 [1] 과 같은 형태 삭제
    roles=["system",'user'] if goals[0]=='Greetings' else ['user','system']
    for i in range(len(kgs)-2): roles.append(roles[i%2])
    return {'uttrs':uttrs, 'roles':roles, 'topics':topics, 'goals':goals, 'kgs':kgs, 'situation':situation, 'user_profile':usr_profile}

def truncationPadding(input_ids, max_length, prefix=[], suffix=[]):
    truncate_size = max_length - len(prefix) - len(suffix)
    # input_ids = prefix + input_ids[-truncate_size:] + suffix
    # input_ids = input_ids + [0] * (max_length - len(input_ids))
    # return input_ids
    if truncate_size <= len(input_ids): input_ids = prefix + input_ids[len(input_ids) - truncate_size:] + suffix
    else: input_ids = prefix + input_ids + suffix
    return input_ids + [0] * (max_length - len(input_ids))

# TODO: 나중에 data loader 를 직접 만들어서 쓸 수도 있을 듯
def batchify(args, batch, tokenizer=None, task=''):
    """
    :param args: args
    :param batch: batch
    :param tokenizer: tokenizer
    :param task: task {'type','topic','know'}
    :return: Tensor[ dialog_token, dialog_mask, response, type, topic, candidate_indice(Optional) ]
    """
    # Input batches are all string
    dialog, user_profile, response, type, topic, situation, target_knowledge = [batch[i] for i in ['dialog', 'user_profile', 'response', 'type', 'topic', 'situation', 'target_knowledge']]
    context_batch = defaultdict()
    suffix_list=[]
    for i in range(len(dialog)): # batch 수 만큼
        suffix = ' '
        if task == 'type': suffix = tokenizer.sep_token
        elif task == 'topic': suffix = tokenizer.sep_token + '<type>' + type[i] + '<user_profile>' + user_profile[i]
        elif task == 'know' :
            if isinstance(topic[i], list): topic[i] = ','.join(topic[i])
            suffix = tokenizer.sep_token + '<situation>' + situation[i] + '<type>' + type[i] + '<topic>' + topic[i]
        else : # Rescponse
            pass
        suffix_list.append(suffix)

    tokenized_dialog = tokenizer(dialog, add_special_tokens=False)
    tokenized_suffix = tokenizer(suffix_list, add_special_tokens=False, max_length=args.max_length//4, truncation=True)

    context_batch['response'] = tokenizer(response, add_special_tokens=True, max_length=args.max_length, padding='max_length', truncation=True).input_ids
    context_batch['dialog_token'] = [truncationPadding(input_ids=dialog_inputids, prefix=[tokenizer.cls_token_id], suffix=suffix_inputids, max_length=args.max_length) for dialog_inputids, suffix_inputids in zip(tokenized_dialog.input_ids, tokenized_suffix.input_ids)]
    context_batch['dialog_mask'] = [truncationPadding(input_ids=dialoginputids, prefix=[1], suffix=suffix_inputids, max_length=args.max_length) for dialoginputids, suffix_inputids in zip(tokenized_dialog.attention_mask, tokenized_suffix.attention_mask)]
    context_batch['type'] = [args.goalDic['str'][i] for i in type]  # index로 바꿈
    context_batch['topic'] = [args.topicDic['str'][i] for i in topic]  # index로 바꿈

    if task == 'know':
        target_knowledge = target_knowledge.tolist()
        candidate_indice = [[know] + negative_sampler(args, know) for know in target_knowledge]
        # candidate_knowledge = tokenizer([args.knowledgeDB[idx] for idx in candidate_indice], truncation=True, padding='max_length', max_length=args.max_length)
        candidate_knowledge_token = [[tokenizer(args.knowledgeDB[i], truncation=True, padding='max_length', max_length=args.max_length).input_ids for i in idx] for idx in candidate_indice]
        candidate_knowledge_mask = [[tokenizer(args.knowledgeDB[i], truncation=True, padding='max_length', max_length=args.max_length).attention_mask for i in idx] for idx in candidate_indice]
        context_batch['candidate_indice'] = candidate_indice  # 이미 Tensor로 받음
        context_batch['candidate_knowledge_token']=candidate_knowledge_token
        context_batch['candidate_knowledge_mask']=candidate_knowledge_mask
        # [target, cand1, cand2, cand3, cand4]

    for k, v in context_batch.items():
        if not isinstance(v, torch.Tensor):
            context_batch[k] = torch.as_tensor(v, device=args.device)
    return context_batch
    # Tensor[dialog_token, dialog_mask, response, type, topic, candidate_indice(Optional)]


def negative_sampler(args, target_knowledge):
    # candidate_entity = self.knowledgeDB[target_knowledge][0]
    # candiate_all_list = self.knowledgeDB_entity_values[candidate_entity]
    # negative_indice = random.choices(candiate_all_list, k=self.args.negative_num if len(candiate_all_list) > self.args.negative_num else len(candiate_all_list))
    total_knowledge_num = args.knowledge_num
    negative_indice = []
    while len(negative_indice) < args.negative_num:
        negative_idx = random.randint(0, total_knowledge_num-1)
        if (negative_idx not in negative_indice) and (negative_idx != target_knowledge):
            negative_indice.append(negative_idx)
    return negative_indice