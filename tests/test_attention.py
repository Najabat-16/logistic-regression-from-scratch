"""Tests for scaled dot-product attention."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.attention import ScaledDotProductAttention, causal_mask

def test_attention_output_shape():
    """Output shape should match (batch, seq_len, d_v)."""
    batch_size, seq_len, d_k, d_v = 2, 4, 8, 8
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_v)
    
    attn = ScaledDotProductAttention(d_k)
    output = attn.forward(Q, K, V)
    
    assert output.shape == (batch_size, seq_len, d_v)

def test_attention_weights_valid():
    """Attention weights should be probabilities (non-negative, sum to 1)."""
    batch_size, seq_len, d_k = 2, 4, 8
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)
    
    attn = ScaledDotProductAttention(d_k)
    attn.forward(Q, K, V)
    
    assert np.all(attn.attention_weights >= 0), "Weights must be non-negative"
    assert np.allclose(attn.attention_weights.sum(axis=-1), 1.0), "Weights must sum to 1"

def test_causal_mask_shape():
    """Causal mask should have correct shape."""
    seq_len = 5
    mask = causal_mask(seq_len)
    assert mask.shape == (seq_len, seq_len)

def test_causal_mask_prevents_future():
    """Causal mask should block future positions."""
    seq_len = 4
    mask = causal_mask(seq_len)
    # Upper triangle should be masked (large negative)
    upper_triangle = np.triu(mask, k=1)
    assert np.all(upper_triangle < -1e9), "Future positions should be masked"

def test_attention_with_mask():
    """Attention should work with causal mask."""
    batch_size, seq_len, d_k, d_v = 1, 4, 8, 8
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_v)
    mask = causal_mask(seq_len)
    
    attn = ScaledDotProductAttention(d_k)
    output = attn.forward(Q, K, V, mask=mask)
    
    assert output.shape == (batch_size, seq_len, d_v)
    assert np.all(np.isfinite(output)), "Output should be finite"

def test_attention_identity_like():
    """With uniform queries/keys, attention should be roughly uniform."""
    batch_size, seq_len, d_k, d_v = 1, 4, 8, 8
    Q = np.ones((batch_size, seq_len, d_k))
    K = np.ones((batch_size, seq_len, d_k))
    V = np.eye(seq_len).reshape(1, seq_len, seq_len)
    
    attn = ScaledDotProductAttention(d_k)
    attn.forward(Q, K, V)
    
    # With uniform Q and K, all attention positions should be roughly equal
    expected_weight = 1.0 / seq_len
    assert np.allclose(attn.attention_weights[0], expected_weight, atol=1e-6)
