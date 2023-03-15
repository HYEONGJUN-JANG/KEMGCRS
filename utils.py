import argparse
import pickle
import os
from datetime import datetime
from pytz import timezone


def get_time_kst(): return datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d_%H%M%S')


def checkPath(path: str) -> None:
    if not os.path.exists(path): os.makedirs(path)


def write_pkl(obj: object, filename: str):
    with open(filename, 'wb') as f: pickle.dump(obj, f)


def read_pkl(filename: str) -> object:
    with open(filename, 'rb') as f: return pickle.load(f)


def parseargs():
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument("--data_cache", action='store_true', help="Whether to run finetune.")
    parser.add_argument("--model_load", action='store_true', help="Whether to load saved model.")

    parser.add_argument("--data_dir", default='data', type=str, help="The data directory.")
    parser.add_argument('--data_name', default='en_test.txt', type=str, help="dataset name")
    parser.add_argument('--k_DB_name', default='knowledgeDB.txt', type=str, help="knowledge DB file name in data_dir")
    parser.add_argument('--k_idx_name', default='knowledge_index.npy', type=str, help="knowledge index file name in data_dir")

    parser.add_argument('--bert_name', default='bert-base-uncased', type=str, help="BERT Model Name")
    parser.add_argument('--model_name', default='myretriever', type=str, help="BERT Model Name")

    parser.add_argument('--pretrained_model', default='bert_model.pt', type=str, help="Pre-trained Retriever BERT Model Name")

    parser.add_argument('--max_length', default=128, type=int, help="dataset name")
    parser.add_argument('--batch_size', default=2, type=int, help="batch size")

    parser.add_argument('--hidden_size', default=768, type=int, help="hidden size")
    parser.add_argument('--num_epochs', default=10, type=int, help="Number of epoch")
    parser.add_argument("--output_dir", default='output', type=str, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--usekg", action='store_true', help="use know_text for response")  # HJ: Know_text 를 사용하는지 여부
    parser.add_argument("--time", default='', type=str, help="Time for fileName")  # HJ : Log file middle Name
    parser.add_argument("--device", default='0', type=str, help="GPU Device")  # HJ : Log file middle Name
    parser.add_argument('--know_topk', default=3, type=int, help="Number of retrieval know text")  # HJ: Know_text retrieve Top-k
    parser.add_argument('--log_dir', default='logs', type=str, help="logging file directory")  # HJ: log file directory
    parser.add_argument('--model_dir', default='models', type=str, help="saved model directory")  # TH: model file directory

    parser.add_argument('--log_name', default='', type=str, help="log file name")  # HJ: log file name
    args = parser.parse_args()
    args.device = f'cuda:{args.device}' if args.device else "cpu"
    if args.time == '': args.time = get_time_kst()
    from platform import system as sysChecker
    if sysChecker() == 'Linux':
        pass  # HJ KT-server
    elif sysChecker() == "Windows":
        pass  # HJ local
    else:
        print("Check Your Platform Setting");
        exit()
    return args


def save_json(args, filename, saved_jsonlines):
    '''
    Args:
        args: args
        filename: file name (path포함)
        saved_jsonlines: Key-value dictionary ( goal_type(str), topic(str), tf(str), dialog(str), target(str), response(str) predict5(list)
    Returns: None
    '''

    def json2txt(saved_jsonlines: list) -> list:
        txtlines = []
        for js in saved_jsonlines:  # TODO: Movie recommendation, Food recommendation, POI recommendation, Music recommendation, Q&A, Chat about stars
            goal, topic, tf, dialog, targetkg, resp, pred5 = js['goal_type'], js['topic'], js['tf'], js['dialog'], js['target'], js['response'], js["predict5"]
            if goal == 'Movie recommendation' or goal=='Food recommendation' or goal=='POI recommendation' or goal=='Music recommendation' or goal=='Q&A' or goal=='Chat about stars':
                pred_txt = "\n".join(pred5)
                txt = f"\n---------------------------\n[Goal]: {goal}\t[Topic]: {topic[0]}\t[TF]: {tf}\n[Target Know_text]: {targetkg}\n[PRED_KnowText]\n{pred_txt}\n[Dialog]"
                for i in dialog.replace("user :", '|user :').replace("system :", "|system : ").split('|'):
                    txt += f"{i}\n"
                txtlines.append(txt)
        return txtlines

    path = os.path.join(args.data_dir, 'print')
    if not os.path.exists(path): os.makedirs(path)
    file = f'{path}/{filename}.txt'
    txts = json2txt(saved_jsonlines)
    with open(file, 'w', encoding='utf-8') as f:
        for i in range(len(txts)): # TODO: 왜 5000개? (TJ)
            f.write(txts[i])


if __name__ == "__main__":
    parseargs()
