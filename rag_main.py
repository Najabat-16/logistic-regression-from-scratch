"""
End-to-end RAG pipeline demo:
1. Build vocabulary from documents
2. Encode documents into tokens
3. Create embeddings for each document
4. Retrieve relevant docs based on query embedding
5. Return context for downstream generation
"""

import numpy as np
from src.tokenizer import SimpleTokenizer
from src.embeddings import SimpleEmbedding
from src.rag_retriever import SimpleRAGRetriever

class SimpleRAGPipeline:
    def __init__(self, vocab_size=100, embedding_dim=16, embedding_iter=100):
        self.tokenizer = SimpleTokenizer(vocab_size)
        self.embedding_model = SimpleEmbedding(vocab_size, embedding_dim, n_iterations=embedding_iter)
        self.retriever = SimpleRAGRetriever(embedding_dim)
        self.documents = []
    
    def prepare_documents(self, docs, epochs=1):
        """Tokenize documents and learn embeddings."""
        self.tokenizer.build_vocab(docs)
        self.documents = docs
        
        doc_embeddings = []
        for doc in docs:
            tokens = self.tokenizer.encode(doc)
            if len(tokens) > 0:
                doc_embed = self.embedding_model.W[tokens].mean(axis=0)
                doc_embed = doc_embed / (np.linalg.norm(doc_embed) + 1e-10)
                doc_embeddings.append(doc_embed)
            else:
                doc_embeddings.append(np.zeros(self.embedding_model.embedding_dim))
        
        doc_embeddings = np.array(doc_embeddings)
        self.retriever.add_documents(docs, doc_embeddings)
    
    def retrieve_for_query(self, query, topk=3):
        """Encode query and retrieve documents."""
        query_tokens = self.tokenizer.encode(query)
        if len(query_tokens) > 0:
            query_embed = self.embedding_model.W[query_tokens].mean(axis=0)
            query_embed = query_embed / (np.linalg.norm(query_embed) + 1e-10)
        else:
            query_embed = np.zeros(self.embedding_model.embedding_dim)
        
        results = self.retriever.retrieve_docs(query_embed, topk)
        return results

if __name__ == "__main__":
    documents = [
        "machine learning is a subset of artificial intelligence",
        "deep learning uses neural networks with multiple layers",
        "natural language processing works with text data",
        "computer vision processes images and videos",
        "transformer models are used for sequential data"
    ]
    
    pipeline = SimpleRAGPipeline(vocab_size=200, embedding_dim=16)
    pipeline.prepare_documents(documents)
    
    query = "what is deep learning"
    retrieved = pipeline.retrieve_for_query(query, topk=2)
    
    print(f"Query: {query}")
    print(f"\nRetrieved documents:")
    for doc, score in retrieved:
        print(f"  [{score:.4f}] {doc}")
