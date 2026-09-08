"""Tests for multi-head attention."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.multihead_attention import MultiHeadAttention

def test_output_shape():
    """Output should match input shape (batch, seq_len, d_model)."""
    mha = MultiHeadAttention(d_model=64, num_heads=8)
    Q = np.random.randn(2, 10, 64)
    K = np.random.randn(2, 10, 64)
    V = np.random.randn(2, 10, 64)
    output = mha.forward(Q, K, V)
    assert output.shape == (2, 10, 64)

def test_attention_weights_shape():
    """Attention weights should be (batch, num_heads, seq_len, seq_len)."""
    mha = MultiHeadAttention(d_model=64, num_heads=8)
    Q = np.random.randn(2, 10, 64)
    K = np.random.randn(2, 10, 64)
    V = np.random.randn(2, 10, 64)
    mha.forward(Q, K, V)
    assert mha.attention_weights.shape == (2, 8, 10, 10)

def test_attention_weights_valid():
    """Attention weights should be valid probabilities."""
    mha = MultiHeadAttention(d_model=64, num_heads=8)
    Q = np.random.randn(2, 10, 64)
    K = np.random.randn(2, 10, 64)
    V = np.random.randn(2, 10, 64)
    mha.forward(Q, K, V)
    assert np.all(mha.attention_weights >= 0)
    assert np.allclose(mha.attention_weights.sum(axis=-1), 1.0)

def test_different_num_heads():
    """Should work with various head counts."""
    for num_heads in [1, 2, 4, 8]:
        mha = MultiHeadAttention(d_model=64, num_heads=num_heads)
        Q = np.random.randn(2, 10, 64)
        K = np.random.randn(2, 10, 64)
        V = np.random.randn(2, 10, 64)
        output = mha.forward(Q, K, V)
        assert output.shape == (2, 10, 64)

def test_split_combine_heads():
    """Split and combine heads should be inverses."""
    mha = MultiHeadAttention(d_model=64, num_heads=8)
    x = np.random.randn(2, 10, 64)
    split = mha.split_heads(x)
    combined = mha.combine_heads(split)
    assert np.allclose(x, combined)

def test_head_dimension():
    """Each head should have dimension d_model / num_heads."""
    d_model = 64
    num_heads = 8
    mha = MultiHeadAttention(d_model, num_heads)
    assert mha.d_k == d_model // num_heads

def test_projection_matrices():
    """Should have Q, K, V, O projection matrices."""
    mha = MultiHeadAttention(d_model=64, num_heads=8)
    assert mha.W_Q.shape == (64, 64)
    assert mha.W_K.shape == (64, 64)
    assert mha.W_V.shape == (64, 64)
    assert mha.W_O.shape == (64, 64)
