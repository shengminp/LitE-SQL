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

SEED = 1996
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def compute_embedding(tokenized_inputs, model, device):
    def last_token_pool(last_hidden_states: Tensor,
                     attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    tokenized_inputs = {
        k: (v.unsqueeze(0) if v.dim() == 1 else v).to(device)
        for k, v in tokenized_inputs.items()
    }
    
    outputs = model(**tokenized_inputs)
    embeddings = last_token_pool(outputs.last_hidden_state, tokenized_inputs['attention_mask'])    
    
    return embeddings


def HardNegativeSuperConLoss(query_embedding, positive_embedding, negative_embedding_list, 
                             temperature=0.07, hard_negative_threshold=0.1, too_hard_negative=True):
    def _l2norm(x):
        return F.normalize(x, p=2, dim=-1)
        
    q = _l2norm(query_embedding)
    d_pos = _l2norm(positive_embedding)
    d_negs = [_l2norm(neg) for neg in negative_embedding_list]
    
    B, _ = q.shape
    
    # (a) positive
    sim_pos = (q * d_pos).sum(dim=-1)
    
    # (b) q_i vs hard negative
    sim_neg_list = []

    for i in range(B):
        if too_hard_negative:
            sim_neg_i = (q[i] * d_negs[i]).sum(dim=-1)
            keep = sim_neg_i >= (sim_pos[i].detach() - hard_negative_threshold)

            if keep.any():
                sim_neg_list.append(torch.logsumexp(sim_neg_i[keep] / temperature, dim=0))
            else:
                sim_neg_list.append(torch.max(sim_neg_i) / temperature)
                
        else:
            sim_neg_i = (q[i] * d_negs[i] / temperature).sum(dim=-1)
            sim_neg_list.append(torch.logsumexp(sim_neg_i, dim=0))

    sim_negs = torch.stack(sim_neg_list)

    sim_pos = sim_pos / temperature

    Z_stack = torch.stack([sim_pos, sim_negs], dim=0)
    
    return (torch.logsumexp(Z_stack, dim=0) - sim_pos).mean()

def calculate_gradient_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

os.system('clear')

device = torch.device('cuda')

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--ft_path', type=str, default='./schema_retriever/language_model/saved_model')
parser.add_argument('--model_name', type=str, default='fine-tuned-embedding-model')
parser.add_argument('--epoch', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_limit', type=int, default=7)
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--temperature', type=float, default=0.07)
parser.add_argument('--LM_MODEL', type=str, default='Qwen/Qwen3-Embedding-0.6B')
parser.add_argument('--DATA_PATH', type=str, default='./schema_retriever/data/fine-tuning_samples_from_BIRD_augmented_version.json')
parser.add_argument('--too_hard_negative', action='store_true')
parser.add_argument('--hard_negative_threshold', type=float, default=0.1)

logger = get_logger()

opt = parser.parse_args()
logger.info("## ARGUMENT INFORMATION ##")
for _ in vars(opt):
    logger.info(f"{_}: {vars(opt)[_]}")
logger.info("##########################")

def main():
    tokenizer = AutoTokenizer.from_pretrained(opt.LM_MODEL, torch_dtype='auto', padding_side = "left")
    model = AutoModel.from_pretrained(opt.LM_MODEL, torch_dtype='auto')

    pad_id = tokenizer.pad_token_id
        
    model.to(device)

    batch_size = opt.batch_size

    dataset = SchemaDataset(
        JSON_DATA_PATH=opt.DATA_PATH, 
        tokenizer=tokenizer,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            collate_fn=lambda batch: collate_fn(batch, pad_id=pad_id, n_negatives_limit=opt.n_limit))
    
    num_epochs = opt.epoch
    num_training_steps = num_epochs * len(dataloader)
    progress_bar = tqdm(range(num_training_steps))

    optimizer = AdamW(model.parameters(), lr=opt.lr)

    model.train()

    folder_path = get_foldername(opt.ft_path)
    os.makedirs(folder_path, exist_ok=True)
    logger.info(f"📢 {opt.model_name} model will be saved on.. {folder_path}")
    
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            query_emb = compute_embedding(batch['query'], model, device)
            pos_emb = compute_embedding(batch['positive'], model, device)
            neg_embs_list = [
                compute_embedding(negatives, model, device) 
                for negatives in batch['negative']
                ]

            loss = HardNegativeSuperConLoss(
                query_embedding=query_emb,
                positive_embedding=pos_emb,
                negative_embedding_list=neg_embs_list,
                hard_negative_threshold=opt.hard_negative_threshold,
                too_hard_negative=opt.too_hard_negative,
                temperature=opt.temperature,
            )
            
            loss.backward()

            total_norm = calculate_gradient_norm(model)
            
            optimizer.step()
            optimizer.zero_grad()
            
            progress_bar.update(1)
            
            if i%100==0:
                logger.info(f"[Epoch {epoch}| Step {i}] Loss: {loss.item():.6f} Gradient Norm: {total_norm:.4f}")
                
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
