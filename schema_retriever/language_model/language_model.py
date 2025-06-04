import torch
from langchain.embeddings.base import Embeddings

class CustomEmbeddings(Embeddings):
    def __init__(self, model, tokenizer, device):
        self.model = model.cuda()
        self.tokenizer = tokenizer
        self.device = device

    def embed_documents(self, texts):
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text):
        return self._get_embedding(text)
    
    def _get_embedding(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        return embedding.tolist()
