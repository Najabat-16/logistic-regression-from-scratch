"""Tests for RAG retriever."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.rag_retriever import SimpleRAGRetriever, RAGPipeline

def test_add_documents():
    """Should add documents correctly."""
    retriever = SimpleRAGRetriever(embedding_dim=16)
    docs = ["doc1", "doc2", "doc3"]
    embeddings = np.random.randn(3, 16)
    retriever.add_documents(docs, embeddings)
    assert len(retriever.documents) == 3
    assert retriever.embeddings.shape == (3, 16)

def test_retrieve_returns_topk():
    """Should return exactly topk results."""
    retriever = SimpleRAGRetriever(embedding_dim=16)
    docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    embeddings = np.random.randn(5, 16)
    retriever.add_documents(docs, embeddings)
    query = np.random.randn(16)
    results = retriever.retrieve(query, topk=3)
    assert len(results) == 3

def test_retrieve_docs_format():
    """Retrieved docs should be (text, score) tuples."""
    retriever = SimpleRAGRetriever(embedding_dim=16)
    docs = ["doc1", "doc2", "doc3"]
    embeddings = np.random.randn(3, 16)
    retriever.add_documents(docs, embeddings)
    query = np.random.randn(16)
    results = retriever.retrieve_docs(query, topk=2)
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    assert all(isinstance(r[0], str) and isinstance(r[1], float) for r in results)

def test_retrieve_topk_capped():
    """topk should be capped by document count."""
    retriever = SimpleRAGRetriever(embedding_dim=16)
    docs = ["doc1", "doc2"]
    embeddings = np.random.randn(2, 16)
    retriever.add_documents(docs, embeddings)
    query = np.random.randn(16)
    results = retriever.retrieve(query, topk=10)
    assert len(results) == 2, "Should return only 2 docs, not requested 10"

def test_rag_pipeline_forward():
    """RAG pipeline should return context and docs."""
    retriever = SimpleRAGRetriever(embedding_dim=16)
    docs = ["machine learning", "deep learning", "nlp"]
    embeddings = np.random.randn(3, 16)
    retriever.add_documents(docs, embeddings)
    pipeline = RAGPipeline(retriever)
    query = np.random.randn(16)
    result = pipeline.forward(query, topk=2)
    assert "context" in result
    assert "retrieved_docs" in result
    assert len(result["retrieved_docs"]) == 2
