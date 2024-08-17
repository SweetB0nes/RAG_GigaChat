import pickle
import torch 

class EmbeddingDatabase:
    def __init__(self):
        self.embeddings = []
        self.data_chunks = []

    def add_embeddings(self, embeddings, chunks):
        self.embeddings.extend(embeddings)
        self.data_chunks.extend(chunks)
    
    def save_to_file(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump((self.embeddings, self.data_chunks), file)
    
    def load_from_file(self, file_path):
        with open(file_path, 'rb') as file:
            self.embeddings, self.data_chunks = pickle.load(file)
    
    def find_top_k(self, query_embedding, k=5):
        similarities = [torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), emb.unsqueeze(0)).item() for emb in self.embeddings]
        ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
        return ranked_indices[:k]
