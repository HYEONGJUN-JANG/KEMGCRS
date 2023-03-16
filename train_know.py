from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim
from utils import *
from models import *

def knowledge_reindexing(args, knowledge_data, retriever):
    print('...knowledge indexing...')
    knowledgeDataLoader = DataLoader(
        knowledge_data,
        batch_size=args.batch_size
    )
    knowledge_index = []

    for batch in tqdm(knowledgeDataLoader):
        input_ids = batch[0].to(args.device)
        attention_mask = batch[1].to(args.device)
        knowledge_emb = retriever.key_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        knowledge_index.extend(knowledge_emb.cpu().detach())
    knowledge_index = torch.stack(knowledge_index, 0)
    return knowledge_index


def update_moving_average(ma_model, current_model):
    decay = 0.99
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = decay * old_weight + (1 - decay) * up_weight


def train_retriever_idx(args, train_dataloader, knowledge_data, retriever):
    # For training BERT indexing
    # train_dataloader = data_pre.dataset_reader(args, tokenizer, knowledgeDB)
    # knowledge_index = knowledge_index.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(retriever.parameters(), lr=args.lr)
    for epoch in range(args.num_epochs):
        knowledge_index = knowledge_reindexing(args, knowledge_data, retriever)
        knowledge_index = knowledge_index.to(args.device)

        total_loss = 0
        for batch in tqdm(train_dataloader):
            dialog_token, dialog_mask, target_knowledge, goal_type, response, topic = batch
            batch_size = dialog_token.size(0)
            dialog_token =dialog_token.to(args.device)
            dialog_mask = dialog_mask.to(args.device)
            target_knowledge = target_knowledge.to(args.device)
            # negative_knowledge = negative_knowledge.to(args.device)


            # tokenizer.batch_decode(dialog_token, skip_special_tokens=True)  # 'dialog context'
            # print([knowledgeDB[idx] for idx in target_knowledge]) # target knowledge

            dot_score = retriever.knowledge_retrieve(dialog_token, dialog_mask, knowledge_index)
            loss = criterion(dot_score, target_knowledge)
            total_loss += loss.data.float()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.momentum:
                update_moving_average(retriever.key_bert, retriever.query_bert)

        print('LOSS:\t%.4f' % total_loss)
    torch.save(retriever.state_dict(), os.path.join(args.model_dir, f"{args.time}_{args.model_name}_bin.pt"))  # TIME_MODELNAME 형식
    return knowledge_index