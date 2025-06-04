import json
import random
import torch
from torch.utils.data import Dataset

N_SAMPLE_LIMIT = 7

class SchemaDataset(Dataset):
    def __init__(self, JSON_DATA_PATH, tokenizer, max_length=256, tags=True):
        with open(JSON_DATA_PATH, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tags = tags

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        query = sample['question']
        sql = sample['SQL']
        evidence = sample.get('evidence', '')
        positive = sample['positive_columns']
        negative_list = sample['negative_columns']
        
        query = f"<question>{query}</question>"
        evidence = f"<evidence>{evidence}</evidence>" if evidence else ''
      
        tokenized_query = self.tokenizer(" ".join([query, evidence] if evidence else query), truncation=True, padding='max_length',
                                         max_length=self.max_length, return_tensors="pt")

        # Query (User quesiton)
        tokenized_query = {k: v.squeeze(0) for k, v in tokenized_query.items()}
        
        # Positive sample
        tokenized_positive = self.tokenizer(positive, truncation=True, padding='max_length',
                                 max_length=self.max_length, return_tensors="pt")
        tokenized_positive = {k: v.squeeze(0) for k, v in tokenized_positive.items()}        
        
        # Negative sample
        tokenized_negative = []
        for neg in negative_list:
            tokens = self.tokenizer(neg, truncation=True, padding='max_length',
                                    max_length=self.max_length, return_tensors="pt")
            tokenized_negative.append({k: v.squeeze(0) for k, v in tokens.items()})
        
        return {
            self.query_type: tokenized_query,
            'positive': tokenized_positive,
            'negative': tokenized_negative
        }

def collate_fn(batch, n_negatives_limit=N_SAMPLE_LIMIT):
    data = {}
    
    for field in batch[0].keys():
        if field == "negative":
            data["negative"] = [
                {
                    key: torch.stack([
                        neg[key] for neg in (
                            random.sample(sample["negative"], n_negatives_limit)
                            if len(sample["negative"]) > n_negatives_limit else sample["negative"]
                        )
                    ])
                    for key in sample["negative"][0].keys()
                }
                for sample in batch
            ]
        else:
            data[field] = {
                key: torch.stack([sample[field][key] for sample in batch])
                for key in batch[0][field]
            }
            
    return data
