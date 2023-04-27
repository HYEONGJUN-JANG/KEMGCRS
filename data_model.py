import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset


def truncationPadding(input_ids, max_length, prefix=[], suffix=[]):
    truncate_size = max_length - len(prefix) - len(suffix)
    # input_ids = prefix + input_ids[-truncate_size:] + suffix
    # input_ids = input_ids + [0] * (max_length - len(input_ids))
    # return input_ids
    if truncate_size <= len(input_ids):
        input_ids = prefix + input_ids[len(input_ids) - truncate_size:] + suffix
    else:
        input_ids = prefix + input_ids + suffix
    return input_ids + [0] * (max_length - len(input_ids))


class GenerationDataset(Dataset):  # knowledge용 데이터셋
    def __init__(self, args, data_sample, knowledgeDB, tokenizer, mode='train', knowledge=False):
        super(Dataset, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.knowledgeDB = knowledgeDB
        self.augmented_raw_sample = data_sample
        self.mode = mode
        self.knowledge = knowledge
        print(knowledge)
        self.generate_prompt_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('System:'))

    def __getitem__(self, idx):  # TODO 구현 전
        data = self.augmented_raw_sample[idx]
        cbdicKeys = ['dialog', 'user_profile', 'response', 'type', 'topic', 'situation', 'target_knowledge']
        dialog, user_profile, response, type, topic, situation, target_knowledge_idx = [data[i] for i in cbdicKeys]
        pad_token_id = self.tokenizer.pad_token_id

        context_batch = defaultdict()
        resp_batch = []
        context_len_batch = []

        prefix = ''
        topic_prompt = self.tokenizer.encode('predict the next response: ')[1:]

        prefix_encoding = self.tokenizer.encode(prefix)[1:][:30]
        knowledge_text = self.knowledgeDB[target_knowledge_idx]

        max_knowledge_length = 30
        if self.knowledge:
            knowledge_text = self.tokenizer('<knowledge>' + self.knowledgeDB[target_knowledge_idx], max_length=max_knowledge_length,
                                            truncation=True).input_ids
        else:
            knowledge_text = []

        dialog = self.tokenizer('<dialog>' + dialog, max_length=self.args.max_length - len(knowledge_text),
                                truncation=True).input_ids
        dialog = knowledge_text + dialog

        if self.mode == 'train':
            response = self.tokenizer(response, max_length=self.args.max_gen_length,
                                      truncation=True).input_ids

            # self.tokenizer.padding_side = 'right'
            max_length = self.args.max_length + self.args.max_gen_length
            context_ids = dialog + response
            context_ids = context_ids[-max_length:]
            context_ids = context_ids + [pad_token_id] * (max_length - len(context_ids))
            # resp_batch = [token_id if token_id != self.tokenizer.pad_token_id else -100 for token_id in context_ids]
            resp_batch = context_ids

            context_batch['input_ids'] = torch.LongTensor(context_ids)
            context_batch['attention_mask'] = torch.ne(context_batch['input_ids'], pad_token_id)
            context_batch['response'] = torch.LongTensor(resp_batch)

        elif self.mode == 'test':
            # self.tokenizer.padding_side = 'left'

            context_ids = dialog[-(self.args.max_length - len(self.generate_prompt_ids)):]
            context_len_batch = len([token for token in context_ids if token != pad_token_id])
            context_ids += self.generate_prompt_ids

            context_ids = [pad_token_id] * (self.args.max_length - len(context_ids)) + context_ids
            context_batch['input_ids'] = torch.LongTensor(context_ids)
            context_batch['attention_mask'] = torch.ne(context_batch['input_ids'], pad_token_id)

            context_batch['response'] = response
            context_batch['context_len'] = context_len_batch

        # for k, v in context_batch.items():
        #     if not isinstance(v, torch.Tensor):
        #         context_batch[k] = torch.as_tensor(v, device=self.args.device)
        # context_batch[k] = torch.as_tensor(v)
        return context_batch

    def __len__(self):
        return len(self.augmented_raw_sample)


def convert_idx_to_docid(idx):
    return f"{idx}"


class DialogDataset(Dataset):  # knowledge용 데이터셋
    def __init__(self, args, data_sample, knowledgeDB, tokenizer, task, mode='train'):
        super(Dataset, self).__init__()
        self.args = args
        self.task = task
        self.tokenizer = tokenizer
        self.knowledgeDB = knowledgeDB
        self.augmented_raw_sample = data_sample
        self.mode = mode

    def negative_sampler(self, target_knowledge):
        # candidate_entity = self.knowledgeDB[target_knowledge][0]
        # candiate_all_list = self.knowledgeDB_entity_values[candidate_entity]
        # negative_indice = random.choices(candiate_all_list, k=self.args.negative_num if len(candiate_all_list) > self.args.negative_num else len(candiate_all_list))
        total_knowledge_num = self.args.knowledge_num
        # negative_indice = list(range(total_knowledge_num))
        # negative_indice = list(set(negative_indice)-set(candidate_positives_idx))
        negative_indice = []
        while len(negative_indice) < self.args.negative_num:
            negative_idx = random.randint(0, total_knowledge_num - 1)
            if (negative_idx not in negative_indice) and (negative_idx != target_knowledge):
                negative_indice.append(negative_idx)
        return negative_indice

    def __getitem__(self, idx):  # TODO 구현 전
        data = self.augmented_raw_sample[idx]
        cbdicKeys = ['dialog', 'user_profile', 'response', 'type', 'topic', 'situation', 'target_knowledge', 'candidate_knowledges', 'candidate_confidences']
        dialog, user_profile, response, type, topic, situation, target_knowledge_idx, candidate_knowledges, candidate_confidences = [data[i] for i in cbdicKeys]
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        context_batch = defaultdict()
        if self.args.input_prompt == 'dialog':
            prefix = ''
        elif self.args.input_prompt == 'dialog_goal':
            prefix = '<type>' + type + self.tokenizer.sep_token
        elif self.args.input_prompt == 'dialog_topic':
            prefix = '<topic>' + topic + self.tokenizer.sep_token
        elif self.args.input_prompt == 'dialog_goal_topic':
            prefix = '<type>' + type + '<topic>' + topic + self.tokenizer.sep_token
        else:
            assert Exception

        prefix_encoding = self.tokenizer.encode(prefix)[1:-1][:30]
        input_sentence = self.tokenizer('<dialog>' + response, add_special_tokens=False).input_ids

        input_sentence = [self.tokenizer.cls_token_id] + prefix_encoding + input_sentence[-(self.args.max_length - len(prefix_encoding) - 1):]
        input_sentence = input_sentence + [pad_token_id] * (self.args.max_length - len(input_sentence))

        context_batch['input_ids'] = torch.LongTensor(input_sentence).to(self.args.device)
        attention_mask = context_batch['input_ids'].ne(pad_token_id)
        context_batch['attention_mask'] = attention_mask
        context_batch['response'] = self.tokenizer(response,
                                                   add_special_tokens=True,
                                                   max_length=self.args.max_length,
                                                   padding='max_length',
                                                   truncation=True).input_ids

        context_batch['type'] = self.args.goalDic['str'][type]  # index로 바꿈
        context_batch['topic_idx'] = self.args.topicDic['str'][topic]  # index로 바꿈
        context_batch['topic'] = self.tokenizer(topic, truncation=True, padding='max_length', max_length=32).input_ids

        # random_idx = random.randrange(min(self.args.pseudo_pos_num, len(candidate_knowledges)))
        # candidate_knowledges = [candidate_knowledges[random_idx]]
        # candidate_confidences = [candidate_confidences[random_idx]]

        # candidate_knowledges = candidate_knowledges[:self.args.pseudo_pos_num]
        # candidate_confidences = candidate_confidences[:self.args.pseudo_pos_num]
        # candidate_knowledges = candidate_knowledges + [0] * (self.args.pseudo_pos_num - len(candidate_knowledges))
        # candidate_confidences = candidate_confidences + [0] * (self.args.pseudo_pos_num - len(candidate_confidences))

        group_num = min(1, len(candidate_knowledges))
        random_idx = random.sample(list(range(min(1, self.args.pseudo_pos_num, len(candidate_knowledges)))), k=group_num)
        candidate_knowledges = [candidate_knowledges[0]]+[candidate_knowledges[idx] for idx in random_idx]
        candidate_confidences = [candidate_confidences[0]] + [candidate_confidences[idx] for idx in random_idx]

        # sampled_pair = sorted(random.sample(list(range(len(candidate_positives_idx))), k=2))
        # pseudo_positive = candidate_positives_idx[sampled_pair[0]]
        # pseudo_negative = candidate_positives_idx[sampled_pair[1]]

        # pseudo_positive = random.choice(candidate_positives_idx)
        # pseudo_positive = candidate_positives_idx[self.args.pseudo_pos_rank - 1]
        pseudo_negative = self.negative_sampler(candidate_knowledges)

        candidate_indice = candidate_knowledges + pseudo_negative  # [candidate_positives_idx[self.args.pseudo_pos_rank]]

        candidate_knowledge_text = [self.args.knowledgeDB[idx] for idx in candidate_indice]
        candidate_knowledge = self.tokenizer(candidate_knowledge_text, truncation=True, padding='max_length', max_length=self.args.max_length)
        candidate_knowledge_token = candidate_knowledge.input_ids
        candidate_knowledge_mask = candidate_knowledge.attention_mask
        #
        context_batch['candidate_indice'] = candidate_indice
        context_batch['candidate_knowledge_token'] = candidate_knowledge_token
        context_batch['candidate_knowledge_mask'] = candidate_knowledge_mask

        context_batch['pseudo_targets'] = candidate_knowledges  # [candidate_knowledges[0]]
        context_batch['pseudo_confidences'] = candidate_confidences

        context_batch['target_knowledge'] = target_knowledge_idx

        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.args.device)
                # context_batch[k] = torch.as_tensor(v)
        return context_batch

    def __len__(self):
        return len(self.augmented_raw_sample)


class KnowledgeDataset(Dataset):
    def __init__(self, args, knowledgeDB, tokenizer):
        super(Dataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.knowledgeDB = knowledgeDB
        self.data_samples = []

    def __getitem__(self, item):
        data = self.knowledgeDB[item]
        tokenized_data = self.tokenizer(data,
                                        max_length=self.max_length,
                                        padding='max_length',
                                        truncation=True,
                                        add_special_tokens=True)
        tokens = torch.LongTensor(tokenized_data.input_ids)
        mask = torch.LongTensor(tokenized_data.attention_mask)
        docid = self.tokenizer.encode(convert_idx_to_docid(item), truncation=True, padding='max_length', max_length=10)[1:-1]  # 이미 Tensor로 받음
        docid = torch.LongTensor(docid)
        return tokens, mask, docid

    def __len__(self):
        return len(self.knowledgeDB)
