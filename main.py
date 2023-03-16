import sys
import logging
from transformers import AutoModel, AutoTokenizer
import data
from config import bert_special_tokens_dict
from train_goal_topic import train_topic, train_goal
from utils import *
from models_hj import *
from data_util import readDic
from platform import system as sysChecker

def main():
    # HJ 작업 --> 형준 여기서만 작업
    args = parseargs()
    args.who="HJ"
    args.data_cache = True
    args.batch_size = 4
    args.bert_saved_model_path = os.path.join("cache", args.bert_name)
    if sysChecker() == 'Linux':
        home = '/home/work/CRSTEST/kemgcrs'
        args.data_dir = os.path.join(home,'data')
        args.output_dir =  os.path.join(args.data_dir,'output')
        args.log_dir =  os.path.join(home,'logs')
        args.models =  os.path.join(home,'model_dir')
        args.bert_saved_model_path = os.path.join(home, "cache", args.bert_name)
        args.batch_size = 128
        pass  # HJ KT-server

    checkPath(args.log_dir)
    checkPath(args.model_dir)

    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(args.log_dir, f'{args.time}_{args.log_name + "_"}log.txt'), filemode='a', format='%(asctime)s: %(levelname)s: %(message)s', datefmt='%Y/%m/%d_%p_%I:%M:%S ')
    logging.info('Commend: {}'.format(', '.join(map(str, sys.argv))))

    # Model cached load
    checkPath(args.bert_saved_model_path)
    bert_model1 = AutoModel.from_pretrained(args.bert_name, cache_dir=args.bert_saved_model_path)
    bert_model2 = AutoModel.from_pretrained(args.bert_name, cache_dir=args.bert_saved_model_path)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
    tokenizer.add_special_tokens(bert_special_tokens_dict)  # [TH] add bert special token (<dialog>, <topic> , <type>)
    bert_model1.resize_token_embeddings(len(tokenizer))
    bert_model2.resize_token_embeddings(len(tokenizer))
    args.hidden_size = bert_model1.config.hidden_size  # BERT large 쓸 때 대비

    # Read knowledge DB
    knowledgeDB = data.read_pkl(os.path.join(args.data_dir, args.k_DB_name))  # TODO: verbalize (TH)
    # knowledge_data = KnowledgeDataset(args, knowledgeDB, tokenizer)  # knowledge dataset class

    topicDic, goalDic = readDic(os.path.join(args.data_dir, "topic2id.txt"),"str"), readDic(os.path.join(args.data_dir, "goal2id.txt"),"str")
    train_dataloader = data.dataset_reader(args, tokenizer, knowledgeDB, data_name='train', goal_dict=goalDic, topic_dict=topicDic)
    test_dataloader = data.dataset_reader(args, tokenizer, knowledgeDB, data_name='test', goal_dict=goalDic, topic_dict=topicDic)
    # knowledge_index = torch.tensor(np.load(os.path.join(args.data_dir, args.k_idx_name)))
    # knowledge_index = knowledge_index.to(args.device)

    # TODO: retriever 로 바꿔서 save 와 load
    args.knowledge_num = len(knowledgeDB)
    args.topic_num = len(topicDic)
    args.goal_num = len(goalDic)
    retriever = Retriever(args, bert_model1, bert_model2)
    retriever = retriever.to(args.device)

    # if args.saved_model_path == '':
    #     knowledge_index = train_retriever_idx(args, train_dataloader, knowledge_data, retriever)  # [TH] <topic> 추가됐으니까 재학습
    # else:
    #     retriever.load_state_dict(torch.load(os.path.join(args.model_dir, args.saved_model_path)))

    # eval_know(args, test_dataloader, retriever, knowledge_index, knowledgeDB, tokenizer) # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리
    train_goal(args, train_dataloader,test_dataloader, retriever, goalDic, tokenizer)

if __name__ == "__main__":
    main()
    # logging.info('Commend: {}'.format(', '.join(map(str, sys.argv))))
