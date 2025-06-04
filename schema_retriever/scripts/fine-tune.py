import os
import json
import random
import argparse
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from schema_retriever.utils.utils import get_logger, get_foldername
from schema_retriever.utils.dataset_util import SchemaDataset, collate_fn


def compute_embedding(tokenized_inputs, model, device):
    tokenized_inputs = {
        k: (v.unsqueeze(0) if v.dim() == 1 else v).to(device)
        for k, v in tokenized_inputs.items()
    }
    outputs = model(**tokenized_inputs)
    embedding = outputs.last_hidden_state[:, 0, :]
    return embedding

def SupConLoss(query_embedding, positive_embedding, negative_embedding_list, temperature=0.07):
    losses = []
    B = query_embedding.size(0)
    
    for i in range(B):
        q = query_embedding[i]      # [D]
        pos = positive_embedding[i] # [D]
        neg = negative_embedding_list[i]  # [N_i, D]

        sim_pos = F.cosine_similarity(q.unsqueeze(0), pos.unsqueeze(0), dim=-1)  # [1]       
        sim_neg = F.cosine_similarity(q.unsqueeze(0).expand(neg.size(0), -1), neg, dim=-1)  # [N_i]
      
        logits = torch.cat([sim_pos, sim_neg], dim=0)  # [1 + N_i]
        label = torch.tensor([0], dtype=torch.long, device=q.device)
        
        loss_i = F.cross_entropy(logits.unsqueeze(0) / temperature, label)
        losses.append(loss_i)
    
    return torch.mean(torch.stack(losses))

os.system('clear')

device = torch.device('cuda')

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--ft_path', type=str, default='./schema_retriever/language_model/saved_model')
parser.add_argument('--model_name', type=str, default='fine-tuned-embedding-model')
parser.add_argument('--epoch', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_limit', type=int, default=7)
parser.add_argument('--LM_MODEL', type=str, default='intfloat/multilingual-e5-large')
parser.add_argument('--DATA_PATH', type=str, default='./schema_retriever/data/fine-tuning_samples_from_BIRD_augmented_version.json')

logger = get_logger()

opt = parser.parse_args()
logger.info("## ARGUMENT INFORMATION ##")
for _ in vars(opt):
    logger.info(f"{_}: {vars(opt)[_]}")
logger.info("##########################")

def main():
    tokenizer = AutoTokenizer.from_pretrained(opt.LM_MODEL, torch_dtype='auto', padding_side = "left")
    model = AutoModel.from_pretrained(opt.LM_MODEL, torch_dtype='auto')
        
    model.to(device)

    batch_size = opt.batch_size

    dataset = SchemaDataset(
        JSON_DATA_PATH=opt.DATA_PATH, 
        tokenizer=tokenizer,
        max_length=256
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            collate_fn=lambda batch: collate_fn(batch, n_negatives_limit=opt.n_limit))
    
    num_epochs = opt.epoch
    num_training_steps = num_epochs * len(dataloader)
    progress_bar = tqdm(range(num_training_steps))

    optimizer = AdamW(model.parameters(), lr=1e-5)

    model.train()

    folder_path = get_foldername(opt.ft_path)
    os.makedirs(folder_path, exist_ok=True)
    logger.info(f"📢 {opt.model_name} model will be saved on.. {folder_path}")
    
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            query_emb = compute_embedding(batch[opt.query_type], model, device)
            pos_emb = compute_embedding(batch['positive'], model, device)
            neg_embs_list = [
                compute_embedding(negatives, model, device) 
                for negatives in batch['negative']
            ]

            loss = SupConLoss(
                query_embedding=query_emb,
                positive_embedding=pos_emb,
                negative_embedding_list=neg_embs_list
            )
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            progress_bar.update(1)
            
            if i%100==0:
                logger.info(f"[Epoch {epoch}| Step {i}] Loss: {loss.item():.6f}")
                
        save_directory = os.path.join(folder_path, f"{opt.model_name}_Epoch-{epoch:02}")

        model.save_pretrained(save_directory)

        tokenizer.save_pretrained(save_directory)            
            
if __name__ == '__main__':
    try:
        main()
    except:
        logger.exception("ERROR!!")
    else:
        logger.handlers.clear()            
