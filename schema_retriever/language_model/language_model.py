import torch
from langchain.embeddings.base import Embeddings

class CustomEmbeddings(Embeddings):
    def __init__(self, model, tokenizer, device, pooling=True):
        self.model = model.cuda()
        self.tokenizer = tokenizer
        self.device = device
        self.pooling = pooling

    def embed_documents(self, texts):
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text):
        return self._get_embedding(text)
    
    def _last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def _get_embedding(self, text):
        inputs = self.tokenizer(text, truncation=True, return_tensors="pt").to(self.device)
       
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        if self.pooling:
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        else:
            embedding = self._last_token_pool(outputs.last_hidden_state, inputs["attention_mask"]).squeeze().to(torch.float32).cpu().numpy()

        return embedding.tolist()
