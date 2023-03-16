from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim
from utils import *
from models import *




def update_moving_average(ma_model, current_model):
    decay = 0.9
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
        print(f"[Epoch-{epoch}]")
        total_loss = 0
        for batch in tqdm(train_dataloader):
            dialog_token, dialog_mask, target_knowledge, goal_type, response, topic, candidate_knowledge_token, candidate_knowledge_mask = batch
            batch_size = dialog_token.size(0)
            dialog_token =dialog_token.to(args.device)
            dialog_mask = dialog_mask.to(args.device)
            target_knowledge = target_knowledge.to(args.device)
            candidate_knowledge_token = candidate_knowledge_token.to(args.device)
            candidate_knowledge_mask = candidate_knowledge_mask.to(args.device)

            # tokenizer.batch_decode(dialog_token, skip_special_tokens=True)  # 'dialog context'
            # print([knowledgeDB[idx] for idx in target_knowledge]) # target knowledge

            logit = retriever.knowledge_retrieve(dialog_token, dialog_mask, candidate_knowledge_token, candidate_knowledge_mask)
            loss = (-torch.log_softmax(logit, dim=1).select(dim=1, index=0)).mean()
            # loss = criterion(dot_score, target_knowledge)
            total_loss += loss.data.float()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.momentum:
                update_moving_average(retriever.key_bert, retriever.query_bert)

        print('LOSS:\t%.4f' % total_loss)

    torch.save(retriever.state_dict(), os.path.join(args.model_dir, f"{args.time}_{args.model_name}_bin.pt"))  # TIME_MODELNAME 형식
