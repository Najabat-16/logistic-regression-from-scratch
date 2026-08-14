"""Tests for SimpleEmbedding model."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.embeddings import SimpleEmbedding

def test_embedding_shape():
    """Embedding matrix should have correct shape."""
    vocab_size = 100
    embedding_dim = 16
    model = SimpleEmbedding(vocab_size, embedding_dim)
    assert model.W.shape == (vocab_size, embedding_dim)
    assert model.V.shape == (embedding_dim, vocab_size)

def test_loss_decreases():
    """Loss should decrease during training."""
    np.random.seed(42)
    vocab_size = 100
    embedding_dim = 16
    center_indices = np.random.randint(0, vocab_size, 500)
    context_indices = np.random.randint(0, vocab_size, 500)
    
    model = SimpleEmbedding(vocab_size, embedding_dim, learning_rate=0.1, n_iterations=50)
    model.fit(center_indices, context_indices)
    
    assert model.loss_history[-1] < model.loss_history[0], "Loss should decrease"

def test_get_embedding():
    """Should return correct embedding vector."""
    vocab_size = 50
    embedding_dim = 8
    model = SimpleEmbedding(vocab_size, embedding_dim)
    embed = model.get_embedding(0)
    assert embed.shape == (embedding_dim,)

def test_most_similar_shape():
    """most_similar should return topk indices and scores."""
    vocab_size = 100
    embedding_dim = 16
    model = SimpleEmbedding(vocab_size, embedding_dim)
    indices, scores = model.most_similar(0, topk=5)
    assert len(indices) == 5
    assert len(scores) == 5

def test_embedding_in_range():
    """Embeddings should be roughly unit norm after initialization."""
    vocab_size = 100
    embedding_dim = 16
    model = SimpleEmbedding(vocab_size, embedding_dim)
    norms = np.linalg.norm(model.W, axis=1)
    assert np.all(norms < 1.0), "Initial embeddings should be small"
