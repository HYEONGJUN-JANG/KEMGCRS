import argparse
import pickle
import os
from datetime import datetime
from pytz import timezone
import numpy as np


def get_time_kst(): return datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d_%H%M%S')


def checkPath(path: str) -> None:
    if not os.path.exists(path): os.makedirs(path)


def write_pkl(obj: object, filename: str):
    with open(filename, 'wb') as f: pickle.dump(obj, f)


def read_pkl(filename: str) -> object:
    with open(filename, 'rb') as f: return pickle.load(f)


def parseargs():
    parser = argparse.ArgumentParser(description="main.py")
    parser.add_argument("--data_cache", action='store_true', help="Whether to run finetune.")
    parser.add_argument("--model_load", action='store_true', help="Whether to load saved model.")
    parser.add_argument("--momentum", action='store_true', help="Whether to load saved model.")
    parser.add_argument("--do_pipeline", action='store_true', help="Whether to load saved model.")
    parser.add_argument("--do_finetune", action='store_true', help="Whether to load saved model.")
    parser.add_argument("--ft_type", action='store_true', help="Whether to Fine-tune on type.")
    parser.add_argument("--ft_topic", action='store_true', help="Whether to Fine-tune on topic.")
    parser.add_argument("--ft_know", action='store_true', help="Whether to Fine-tune on know.")
    parser.add_argument("--earlystop", action='store_true', help="Whether to Use EarlyStopping.")
    parser.add_argument("--task", default='know', type=str, help="Choose the task")
    parser.add_argument("--knowledge", action='store_true', help="Whether to Use knowledge in response.")
    parser.add_argument("--know_ablation", default='pseudo', type=str, help="know_ablation", choices=['target', 'pseudo'])
    parser.add_argument("--train_ablation", default='learning2rank', type=str, help="train ablation", choices=['learning2rank', 'sampling'])

    parser.add_argument("--pseudo", action='store_true', help="Whether to Fine-tune on type.")
    parser.add_argument('--pseudo_pos_num', default=2, type=int, help="pseudo_pos_num")
    parser.add_argument('--pseudo_pos_rank', default=2, type=int, help="pseudo_pos_rank")
    parser.add_argument("--pseudo_confidence", action='store_true', help="Whether to Fine-tune on type.")
    parser.add_argument('--tau', type=float, default=1.0, help='Learning rate')
    parser.add_argument('--bin', type=int, default=0, help='bin')
    parser.add_argument('--negative_num', default=4, type=int, help="negative_num")
    parser.add_argument('--stage', default='rerank', type=str, choices=['retrieve', 'rerank'])
    parser.add_argument("--stage2_test", action='store_true', help="Whether to Fine-tune on type.")

    parser.add_argument('--update_freq', default=-1, type=int, help="update_freq")

    parser.add_argument("--data_dir", default='data', type=str, help="The data directory.")
    # parser.add_argument('--data_name', default='en_test.txt', type=str, help="dataset name")
    parser.add_argument('--k_DB_name', default='all_knowledge_DB.pickle', type=str, help="knowledge DB file name in data_dir")
    parser.add_argument('--k_idx_name', default='knowledge_index.npy', type=str, help="knowledge index file name in data_dir")

    ## Model BERT or BART
    parser.add_argument("--type_aware", action='store_true', help="Whether to Use Type-aware Matching")
    parser.add_argument('--kencoder_name', default='bert-base-uncased', type=str, help="Knowledge Encoder Model Name")
    parser.add_argument('--qencoder_name', default='facebook/bart-base', type=str, help="Query Encoder Model Name")

    parser.add_argument('--bart_name', default='facebook/bart-base', type=str, help="BERT Model Name")
    parser.add_argument('--bert_name', default='./bert-base-uncased', type=str, help="BERT Model Name")
    parser.add_argument('--gpt_name', default='gpt2', type=str, help="BERT Model Name")

    parser.add_argument('--model_name', default='myretriever', type=str, help="BERT Model Name")

    parser.add_argument('--pretrained_model', default='bert_model.pt', type=str, help="Pre-trained Retriever BERT Model Name")

    parser.add_argument('--max_length', default=15, type=int, help="dataset name")
    parser.add_argument('--batch_size', default=16, type=int, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--lr_rerank', type=float, default=1e-6, help='Learning rate')

    parser.add_argument('--loss_lamb', type=float, default=1.0, help='Learning rate')
    parser.add_argument('--lr_dc_step', type=int, default=10, help='warmup_step')
    parser.add_argument('--lr_dc', type=float, default=0.1, help='warmup_gamma')

    parser.add_argument('--hidden_size', default=768, type=int, help="hidden size")
    parser.add_argument('--num_epochs', default=10, type=int, help="Number of epoch")
    parser.add_argument("--output_dir", default='output', type=str, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--usekg", action='store_true', help="use know_text for response")  # HJ: Know_text 를 사용하는지 여부
    parser.add_argument("--time", default='', type=str, help="Time for fileName")  # HJ : Log file middle Name
    parser.add_argument("--loss_rec", default='cross_entropy', type=str, help="Loss Type")  # HJ : Loss
    parser.add_argument('--lamb', type=float, default=0.8, help='lambda for loss target')

    parser.add_argument("--device", default='0', type=str, help="GPU Device")  # HJ : Log file middle Name

    parser.add_argument('--know_topk', default=20, type=int, help="Number of retrieval know text")  # HJ: Know_text retrieve Top-k
    parser.add_argument('--topic_topk', default=5, type=int, help="Number of Top-k Topics")  # HJ: Topic Top-k
    parser.add_argument('--home', default='', type=str, help="Project home directory")  # HJ: Project Home directory
    parser.add_argument('--log_dir', default='logs', type=str, help="logging file directory")  # HJ: log file directory
    parser.add_argument('--model_dir', default='models', type=str, help="saved model directory")  # TH: model file directory
    parser.add_argument('--saved_model_path', default='', type=str, help="saved model file name")  # TH: model file directory

    parser.add_argument('--log_name', default='', type=str, help="log file name")  # HJ: log file name

    # TH
    parser.add_argument('--retrieve', default='negative', type=str, help="retrieve")
    parser.add_argument('--input_prompt', default='dialog', type=str, help="input_prompt")

    args = parser.parse_args()
    # args.model_dir = os.path.join(args.model_dir, args.device)
    args.device = f'cuda:{args.device}' if args.device else "cpu"
    if args.time == '': args.time = get_time_kst()
    from platform import system as sysChecker
    if sysChecker() == 'Linux':
        # HJ KT-server
        args.home = '/home/work/CRSFolder/KEMGCRS'
        args.data_dir = os.path.join(args.home, 'data')
        args.output_dir = os.path.join(args.data_dir, 'output')
        args.log_dir = os.path.join(args.home, 'logs')
        args.model_dir = os.path.join(args.home, 'models')
        args.bert_saved_model_path = os.path.join(args.home, "cache", args.kencoder_name)
        # args.batch_size = 64
        # args.num_epochs = 25
        pass  # HJ KT-server
    elif sysChecker() == "Windows":
        args.data_cache = True
        # args.batch_size = 4
        args.log_name = "log_Topic PRF"
        # args.task = 'goal'
        args.num_epochs = 2
        args.do_finetune = True
        pass  # HJ local
    else:
        print("Check Your Platform Setting");
        exit()

    args.usebart = True
    args.bert_cache_name = os.path.join(args.home, "cache", args.kencoder_name)

    return args


def save_json(args, filename, saved_jsonlines):
    '''
    Args:
        args: args
        filename: file name (path포함)
        saved_jsonlines: Key-value dictionary ( goal_type(str), topic(str), tf(str), dialog(str), target(str), response(str) predict5(list)
    Returns: None
    '''
    correct_ranking = np.array([0.0] * args.know_topk)
    cnt = [0.0]

    def json2txt(saved_jsonlines: list) -> list:
        txtlines = []
        for js in saved_jsonlines:  # TODO: Movie recommendation, Food recommendation, POI recommendation, Music recommendation, Q&A, Chat about stars
            goal, topic, tf, dialog, targetkg, resp, pred5, score5 = js['goal_type'], js['topic'], js['tf'], js['dialog'], js['target'], js['response'], js["predict5"], js['score5']
            if goal == 'Movie recommendation' or goal == 'POI recommendation' or goal == 'Music recommendation' or goal == 'Q&A' or goal == 'Chat about stars':
                bm_ranking = np.argsort(score5)[::-1]
                for idx in range(len(bm_ranking)):
                    if idx == bm_ranking[idx]:
                        correct_ranking[idx] += 1
                cnt[0] += 1

                pred_text = ["%s(%.4f)" % (p, s) for p, s in zip(pred5, list(score5))]
                pred_txt = "\n".join(pred_text)
                txt = f"\n---------------------------\n[Goal]: {goal}\t[Topic]: {topic}\t[TF]: {tf}\n[Target Know_text]: {targetkg}\n[PRED_KnowText]\n{pred_txt}\n[Dialog]\n"
                for i in dialog.replace("user :", '|user :').replace("system :", "|system : ").split('|'):
                    txt += f"{i}\n"
                txt += f"[Response]: {resp}\n"
                txtlines.append(txt)
        return txtlines

    path = os.path.join(args.data_dir, 'print')
    if not os.path.exists(path): os.makedirs(path)
    file = f'{path}/{filename}.txt'
    txts = json2txt(saved_jsonlines)
    with open(file, 'w', encoding='utf-8') as f:
        for i in range(len(txts)):
            f.write(txts[i])
    print(correct_ranking / cnt[0])


def checkGPU(args, logger=None):
    import torch.cuda
    logger.info('Memory Usage on {}'.format(torch.cuda.get_device_name(device=args.device)))
    logger.info('Allocated: {} GB'.format(round(torch.cuda.memory_allocated(device=args.device) / 1024 ** 3, 1)))
    logger.info('Cached:   {} GB'.format(round(torch.cuda.memory_cached(device=args.device) / 1024 ** 3, 1)))
    return False


if __name__ == "__main__":
    parseargs()
