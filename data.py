from tqdm import tqdm
from torch.utils.data import DataLoader
from dataModel import KnowledgeDataset, DialogDataset
import numpy as np
import json

from data_hj import dataset_reader_raw_hj
from utils import *


def knowledge_db_save(args, knowledgeDB, max_length, tokenizer, model):
    knowledgeDataset = KnowledgeDataset(knowledgeDB, max_length, tokenizer)
    knowledgeDataLoader = DataLoader(
        knowledgeDataset,
        batch_size=2
    )
    model = model.to(args.device)
    knowledge_index = []
    for batch in tqdm(knowledgeDataLoader):
        input_ids = batch[0].to(args.device)
        attention_mask = batch[1].to(args.device)
        knowledge_emb = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        # knowledge_index.append()
        knowledge_index.extend(knowledge_emb.cpu().detach().numpy())
    print('done')

    # np.save('knowledge_index.npy', knowledge_index)
    np.save(os.path.join(args.data_dir, args.k_idx_name), knowledge_index)


def dataset_reader(args, tokenizer, knowledgeDB, data_name='train'):
    if args.who=='TH':
        return dataset_reader_raw(args, tokenizer, knowledgeDB, data_name='train')
    elif args.who=="HJ":
        return dataset_reader_raw_hj(args, tokenizer, knowledgeDB, data_name='train')
    else:
        pass


def dataset_reader_raw(args, tokenizer, knowledgeDB, data_name='train'):
    if not os.path.exists(os.path.join(args.data_dir, 'cache')): os.makedirs(os.path.join(args.data_dir, 'cache'))
    cachename = os.path.join(args.data_dir, 'cache', f"cached_en_{data_name}.pkl")
    cachename_know = os.path.join(args.data_dir, 'cache', f"cached_en_{data_name}_know.pkl")

    if args.data_cache and os.path.exists(cachename) and os.path.exists(cachename_know):
        train_sample = read_pkl(cachename)
        knowledge_sample = read_pkl(cachename_know)
    else:
        train_sample = []
        knowledge_sample = []
        data_path = os.path.join(args.data_dir, f"en_{data_name}.txt")
        with open(data_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f, desc="Dataset Read", bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
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
                        suffix = '<type>' + dialog['goal_type_list'][i] + '<topic>' + dialog['goal_topic_list'][i]  # [TH] 일단 임시로 넣어봄

                        # Truncate and padding
                        tokenized_dialog = tokenizer(flatten_dialog, add_special_tokens=False)
                        tokenized_prefix = tokenizer(suffix, add_special_tokens=False)
                        input_ids = truncationPadding(input_ids=tokenized_dialog.input_ids, prefix=[tokenizer.cls_token_id], suffix=tokenized_prefix.input_ids, max_length=args.max_length)
                        attention_mask = truncationPadding(input_ids=tokenized_dialog.attention_mask, prefix=[1], suffix=tokenized_prefix.attention_mask, max_length=args.max_length)

                        # TODO: argument 받아서 처리하기
                        # tokenized_dialog = tokenizer(flatten_dialog,
                        #                              padding='max_length',
                        #                              truncation=True,
                        #                              add_special_tokens=True)
                        train_sample.append({'dialog_token': input_ids,
                                             'dialog_mask': attention_mask,
                                             'response': conversation[i],
                                             'goal_type': dialog['goal_type_list'][i],
                                             'topic': dialog['goal_topic_list'][i]})
                        if knowledge_seq[i] != '':
                            input_ids = truncationPadding(input_ids=tokenized_dialog.input_ids, suffix=[tokenizer.cls_token_id], max_length=args.max_length)
                            attention_mask = truncationPadding(input_ids=tokenized_dialog.attention_mask, suffix=[1], max_length=args.max_length)
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
    data_sample = DialogDataset(args, knowledge_sample)
    batch_size = args.batch_size if 'train' == data_name else 1
    dataloader = DataLoader(data_sample, batch_size=batch_size)
    return dataloader


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
    dataset_reader(args, tokenizer, knowledgeDB)
