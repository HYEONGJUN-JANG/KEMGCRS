import sys
import logging
from collections import defaultdict

from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import data
from config import bert_special_tokens_dict
from dataModel import KnowledgeDataset, DialogDataset
from eval_know import eval_know
from train_know import train_retriever_idx
from utils import *
from models import *
from data_util import readDic


def main():
    # TH 작업 main
    args = parseargs()
    # args.data_cache = False
    args.who = "TH"

    checkPath(args.log_dir)
    checkPath(args.model_dir)

    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(args.log_dir, f'{args.time}_{args.log_name + "_"}log.txt'), filemode='a', format='%(asctime)s: %(levelname)s: %(message)s', datefmt='%Y/%m/%d_%p_%I:%M:%S ')
    logging.info('Commend: {}'.format(', '.join(map(str, sys.argv))))

    # Model cached load
    checkPath(os.path.join("cache", args.bert_name))

    bert_model1 = AutoModel.from_pretrained(args.bert_name, cache_dir=os.path.join("cache", args.bert_name))
    bert_model2 = AutoModel.from_pretrained(args.bert_name, cache_dir=os.path.join("cache", args.bert_name))

    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
    tokenizer.add_special_tokens(bert_special_tokens_dict)  # [TH] add bert special token (<dialog>, <topic> , <type>)
    bert_model1.resize_token_embeddings(len(tokenizer))
    bert_model2.resize_token_embeddings(len(tokenizer))
    args.hidden_size = bert_model1.config.hidden_size  # BERT large 쓸 때 대비

    # Read knowledge DB
    # train_knowledgeDB = data.read_pkl(os.path.join(args.data_dir, 'train_knowledge_DB.pickle'))  # TODO: verbalize (TH)
    knowledgeDB = data.read_pkl(os.path.join(args.data_dir, 'knowledgeDB.txt'))  # TODO: verbalize (TH)
    knowledge_data = KnowledgeDataset(args, knowledgeDB, tokenizer)  # knowledge dataset class
    args.knowledge_num = len(knowledgeDB)

    all_knowledgeDB = data.read_pkl(os.path.join(args.data_dir, 'all_knowledge_DB.pickle'))  # TODO: verbalize (TH)
    knowledgeDB_values = [k[1] for k in all_knowledgeDB]
    knowledgeDB_entity_values = defaultdict(list)
    for k in all_knowledgeDB:
        knowledgeDB_entity_values[k[0]].append(knowledgeDB_values.index(k[1]))

    train_dataset = data.dataset_reader(args, tokenizer, knowledgeDB, 'train')
    test_dataset = data.dataset_reader(args, tokenizer, knowledgeDB, 'test')

    train_datamodel = DialogDataset(args, train_dataset, knowledgeDB, tokenizer)
    test_datamodel = DialogDataset(args, test_dataset, knowledgeDB, tokenizer)

    train_dataloader = DataLoader(train_datamodel, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_datamodel, batch_size=1, shuffle=False)

    topicDic, goalDic = readDic(os.path.join(args.data_dir, "topic2id.txt"), "idx"), readDic(os.path.join(args.data_dir, "goal2id.txt"), "idx")
    # knowledge_index = torch.tensor(np.load(os.path.join(args.data_dir, args.k_idx_name))).to(args.device)
    # knowledge_index = knowledge_index.to(args.device)

    # TODO: retriever 로 바꿔서 save 와 load
    # if args.model_load:
    #     bert_model.load_state_dict(torch.load(os.path.join(args.model_dir, args.pretrained_model)))  # state_dict를 불러 온 후, 모델에 저장`

    retriever = Retriever(args, bert_model1, bert_model2)
    retriever = retriever.to(args.device)

    if args.saved_model_path == '':
        train_retriever_idx(args, train_dataloader, knowledge_data, retriever)  # [TH] <topic> 추가됐으니까 재학습
    else:
        retriever.load_state_dict(torch.load(os.path.join(args.model_dir, args.saved_model_path)))

    eval_know(args, test_dataloader, retriever, knowledge_data, knowledgeDB, tokenizer)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리


if __name__ == "__main__":
    main()
    # logging.info('Commend: {}'.format(', '.join(map(str, sys.argv))))
