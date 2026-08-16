"""
Simple RAG retriever - retrieve relevant documents based on embedding similarity.
"""

import numpy as np

class SimpleRAGRetriever:
    def __init__(self, embedding_dim=16):
        self.embedding_dim = embedding_dim
        self.documents = []
        self.embeddings = np.array([])
    
    def add_documents(self, docs, doc_embeddings):
        """
        Add documents with their embeddings.
        docs: list of strings
        doc_embeddings: (n_docs, embedding_dim) array
        """
        self.documents = docs
        self.embeddings = np.asarray(doc_embeddings, dtype=np.float64)
        
        if len(self.documents) != len(self.embeddings):
            raise ValueError("Number of docs must match number of embeddings")
    
    def retrieve(self, query_embedding, topk=3):
        """
        Retrieve top-k most similar documents.
        query_embedding: (embedding_dim,) array
        Returns: list of (doc_idx, similarity_score) tuples
        """
        if len(self.documents) == 0:
            return []
        
        query_embedding = np.asarray(query_embedding, dtype=np.float64)
        
        # Cosine similarity: (q @ d) / (||q|| * ||d||)
        query_norm = np.linalg.norm(query_embedding)
        doc_norms = np.linalg.norm(self.embeddings, axis=1)
        
        similarities = (self.embeddings @ query_embedding) / (doc_norms * query_norm + 1e-15)
        
        # Get top-k indices
        topk = min(topk, len(self.documents))
        top_indices = np.argsort(similarities)[-topk:][::-1]
        
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        return results
    
    def retrieve_docs(self, query_embedding, topk=3):
        """Return actual document texts with scores."""
        results = self.retrieve(query_embedding, topk)
        return [(self.documents[idx], score) for idx, score in results]

class RAGPipeline:
    """Simple RAG: retrieve documents, then generate answer."""
    
    def __init__(self, retriever):
        self.retriever = retriever
    
    def retrieve_context(self, query_embedding, topk=3):
        """Get context documents for a query."""
        docs_with_scores = self.retriever.retrieve_docs(query_embedding, topk)
        context = "\n".join([f"[Doc {i}] {doc}" for i, (doc, score) in enumerate(docs_with_scores)])
        return context, docs_with_scores
    
    def forward(self, query_embedding, topk=3):
        """Full RAG forward pass - just retrieve for now."""
        context, docs = self.retrieve_context(query_embedding, topk)
        return {
            "context": context,
            "retrieved_docs": docs
        }
