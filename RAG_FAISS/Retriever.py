from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

class Retriever:
    def __init__(self, documents, model_name="intfloat/e5-large"):
        embed_model = HuggingFaceEmbeddings(model_name=model_name)
        self.db = FAISS.from_documents(documents, embed_model)
    
    def retrieve(self, query, k=5):
        retriever = self.db.as_retriever(search_type='mmr', search_kwargs={
            'k': k,
            'lambda_mult': 0.5
        })
        return retriever.invoke(query)