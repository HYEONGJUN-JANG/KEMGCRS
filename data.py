from tqdm import tqdm
from torch.utils.data import DataLoader
from dataModel import KnowledgeDataset, DialogDataset
import numpy as np
import json
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


def dataset_reader(args, tokenizer, knowledgeDB):
    if not os.path.exists(os.path.join(args.data_dir, 'cache')): os.makedirs(os.path.join(args.data_dir, 'cache'))
    cachename = os.path.join(args.data_dir, 'cache', f"cached_{args.data_name}.pt")
    if args.data_cache and os.path.exists(cachename):
        train_sample = read_pkl(cachename)
    else:
        train_sample = []
        knowledge_sample = []
        data_path = os.path.join(args.data_dir, args.data_name)
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

                        tokenized_dialog = tokenizer(flatten_dialog,
                                                     max_length=args.max_length,
                                                     padding='max_length',
                                                     truncation=True,
                                                     add_special_tokens=True)
                        train_sample.append({'dialog_token': tokenized_dialog.input_ids,
                                             'dialog_mask': tokenized_dialog.attention_mask,
                                             'response': conversation[i],
                                             'goal_type': dialog['goal_type_list'][i],
                                             'topic': dialog['goal_topic_list'][i]
                                             })
                        if knowledge_seq[i] != '':
                            knowledge_sample.append({'dialog_token': tokenized_dialog.input_ids,
                                                     'dialog_mask': tokenized_dialog.attention_mask,
                                                     'response': conversation[i],
                                                     'goal_type': dialog['goal_type_list'][i],
                                                     'topic': dialog['goal_topic_list'][i],
                                                     'target_knowledge': knowledgeDB.index(knowledge_seq[i])
                                                     })
                    augmented_dialog.append(conversation[i])
        if args.data_cache: write_pkl(train_sample, cachename)

    train_sample = DialogDataset(train_sample)
    train_dataloader = DataLoader(train_sample, batch_size=1)
    return train_dataloader


if __name__ == "__main__":
    from transformers import AutoModel, AutoTokenizer

    args = parseargs()
    args.data_cache = False
    if not os.path.exists(os.path.join("cache", args.model_name)): os.makedirs(os.path.join("cache", args.model_name))
    bert_model = AutoModel.from_pretrained(args.model_name, cache_dir=os.path.join("cache", args.model_name))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    knowledgeDB = read_pkl("data/knowledgeDB.txt")
    dataset_reader(args, tokenizer, knowledgeDB)