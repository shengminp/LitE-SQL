import json
import random
import torch
from torch.utils.data import Dataset

SEED = 1996
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

class SchemaDataset(Dataset):
    def __init__(self, JSON_DATA_PATH, tokenizer):
        with open(JSON_DATA_PATH, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        query = sample['question']
        evidence = sample.get('evidence', '')
        positive = sample['positive_columns']
        negative_list = sample['negative_columns']
        
        instruction = "Instruct:Given a natural language question, retrieve database column information passages used to generate SQL.\n"
        
        # add query and passage token -----------------------------------
        query = f"{instruction}Query:{query}"
        evidence = f" {evidence}" if evidence else ''
        # ---------------------------------------------------------------
        
        # Query (User quesiton)
        tokenized_query = self.tokenizer(" ".join([query, evidence] if evidence else query), return_tensors="pt")
        tokenized_query = {k: v.squeeze(0) for k, v in tokenized_query.items()}
        
        # Positive sample
        tokenized_positive = self.tokenizer(positive, truncation=True, return_tensors="pt")
        tokenized_positive = {k: v.squeeze(0) for k, v in tokenized_positive.items()}
        
        # Negative sample
        tokenized_negative = []
        for neg in negative_list:
            tokens = self.tokenizer(neg, truncation=True, return_tensors="pt")            
            tokenized_negative.append({k: v.squeeze(0) for k, v in tokens.items()})
        
        return {
            'query': tokenized_query,
            'positive': tokenized_positive,
            'negative': tokenized_negative
        }

def _left_pad(seq_list, pad_id):
    """
    seq_list: list of 1D Tensors
    pad_value: value used for padding
    Returns a stacked Tensor with left padding.
    """
    
    max_length = max(seq.size(0) for seq in seq_list)
    
    padded_seqs = []
    attention_masks = []
    
    for seq in seq_list:
        pad_size = max_length - seq.size(0)
        
        # Create left padding
        left_padding = torch.full((pad_size,), pad_id, dtype=seq.dtype)
        
        # Concatenate [pad, ..., pad, real_ids]
        padded_seq = torch.cat((left_padding, seq), dim=0)
        padded_seqs.append(padded_seq)

        seq_mask = torch.ones((seq.size(0)), dtype=torch.bool)
        pad_mask = torch.zeros((pad_size), dtype=torch.bool)
        attention_mask = torch.cat((pad_mask, seq_mask), dim=0)
        attention_masks.append(attention_mask)

    return torch.stack(padded_seqs, dim=0), torch.stack(attention_masks, dim=0)

def collate_fn(batch, pad_id, n_negatives_limit=10):
    data = {}
    
    for field in batch[0].keys():
        if field == "negative":
            neg_list = [
                [
                    neg['input_ids'] for neg in (
                    random.sample(sample[field], n_negatives_limit)
                    if len(sample[field]) > n_negatives_limit else sample[field]
                    )
                ]
                for sample in batch
            ]

            data[field] = []
            for neg in neg_list:
                input_id, attn_mask = torch.stack(_left_pad(neg, pad_id=pad_id))
                data[field].append(
                    {
                        "input_ids": input_id,
                        "attention_mask": attn_mask
                    }
                )
        
        else:
            input_id, attn_mask = torch.stack(_left_pad([sample[field]['input_ids'] for sample in batch], pad_id=pad_id))
            data[field] = {
                "input_ids": input_id,
                "attention_mask": attn_mask
            }
            
    return data
