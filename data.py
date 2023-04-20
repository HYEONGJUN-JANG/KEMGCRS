from collections import defaultdict

from tqdm import tqdm
from torch.utils.data import DataLoader
from data_model import KnowledgeDataset
import numpy as np
import json
import data_hj
# from data_hj import dataset_reader_raw_hj, DialogDataset
from utils import *
import logging
logger = logging.getLogger(__name__)

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


def dataset_reader(args, tokenizer, knowledgeDB, mode='train', goal_dict=None, topic_dict=None, pred_dict=None):
    logger.info("Dataset For Task: {}".format(args.task))
    if args.task=='know':
        data_sample = dataset_reader_th(args, tokenizer, knowledgeDB, mode)
        data_datamodel = KnowDialogDataset(args, data_sample, knowledgeDB, tokenizer)
        if mode == 'train': return DataLoader(data_datamodel, batch_size=args.batch_size, shuffle=True) # Train
        else : return DataLoader(data_datamodel, batch_size=1, shuffle=False) # Test
    elif args.task in ['goal','topic', 'goal_pipe','topic_pipe','know_pipe', 'resp_pipe']:
        data_sample = data_hj.dataset_reader_raw_hj(args, tokenizer, knowledgeDB, data_name=mode, goal_dict=goal_dict, topic_dict=topic_dict)
        data_sample = data_hj.DialogDataset(args, data_sample, pred_dict=pred_dict, mode=mode, goal_dict=goal_dict, topic_dict=topic_dict, knowledgeDB=knowledgeDB, tokenizer = tokenizer)
        batch_size = args.batch_size # if 'train' == data_name else 1
        return DataLoader(data_sample, batch_size=batch_size)
    else:
        pass

# HJ: Don't Touch
def dataset_reader_th(args, tokenizer, knowledgeDB, data_name='train'):
    if not os.path.exists(os.path.join(args.data_dir, 'cache')): os.makedirs(os.path.join(args.data_dir, 'cache'))
    cachename = os.path.join(args.data_dir, 'cache', f"cached_en_{data_name}.pkl")
    cachename_know = os.path.join(args.data_dir, 'cache', f"cached_en_{data_name}_know.pkl")

    if args.data_cache and os.path.exists(cachename) and os.path.exists(cachename_know):
        print(f"Read Pickle {cachename}")
        train_sample = read_pkl(cachename)
        knowledge_sample = read_pkl(cachename_know)
    else:
        # train_sample = []
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
                for i in range(len(conversation)):
                    conversation[i] = conversation[i] if conversation[i][0] != '[' else conversation[i][4:]
                    conversation[i] = role_seq[i] + ": " + conversation[i]

                augmented_dialog = []
                for i in range(len(conversation)):
                    role = role_seq[i]
                    if role == 'System' and len(augmented_dialog) > 0:
                        flatten_dialog = tokenizer.sep_token.join(augmented_dialog)
                        if knowledge_seq[i] != '':
                            knowledge_sample.append({'dialog': flatten_dialog,
                                                     'user_profile': user_profile,
                                                     'response': conversation[i],
                                                     'goal_type': dialog['goal_type_list'][i],
                                                     'topic': dialog['goal_topic_list'][i],
                                                     'situation': situation,
                                                     'target_knowledge': knowledgeDB.index(knowledge_seq[i])})
                    augmented_dialog.append(conversation[i])

        if args.data_cache:
            # write_pkl(train_sample, cachename)
            write_pkl(knowledge_sample, cachename_know)

    return knowledge_sample


def truncationPadding(input_ids, max_length, prefix=[], suffix=[]):
    truncate_size = max_length - len(prefix) - len(suffix)
    input_ids = prefix + input_ids[-truncate_size:] + suffix
    input_ids = input_ids + [0] * (max_length - len(input_ids))
    return input_ids

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
    dataset_reader(args, tokenizer, knowledgeDB)
