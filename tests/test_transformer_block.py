"""Tests for transformer block."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.transformer_block import TransformerBlock

def test_output_shape():
    """Output should match input shape."""
    block = TransformerBlock(d_model=64, num_heads=8, d_ff=256)
    x = np.random.randn(2, 10, 64)
    output = block.forward(x)
    assert output.shape == (2, 10, 64)

def test_attention_weights_exist():
    """Attention weights should be stored."""
    block = TransformerBlock(d_model=64, num_heads=8, d_ff=256)
    x = np.random.randn(2, 10, 64)
    block.forward(x)
    assert block.attention_weights is not None
    assert block.attention_weights.shape == (2, 8, 10, 10)

def test_output_bounded():
    """Output should be reasonably bounded (normalized)."""
    block = TransformerBlock(d_model=64, num_heads=8, d_ff=256)
    x = np.random.randn(2, 10, 64)
    output = block.forward(x)
    assert np.all(np.isfinite(output)), "Output should be finite"
    assert np.linalg.norm(output) < 200, "Output should be bounded"

def test_residual_connection_shape():
    """Residual connections should preserve shape."""
    block = TransformerBlock(d_model=64, num_heads=8, d_ff=256)
    x = np.random.randn(2, 10, 64)
    output = block.forward(x)
    assert output.shape == x.shape

def test_stacking_blocks():
    """Multiple transformer blocks should stack correctly."""
    block1 = TransformerBlock(d_model=64, num_heads=8, d_ff=256)
    block2 = TransformerBlock(d_model=64, num_heads=8, d_ff=256)
    x = np.random.randn(2, 10, 64)
    x = block1.forward(x)
    x = block2.forward(x)
    assert x.shape == (2, 10, 64)

def test_layer_norm():
    """Layer normalization should normalize activations."""
    block = TransformerBlock(d_model=64, num_heads=8, d_ff=256)
    x = np.random.randn(2, 10, 64)
    x_normalized = block.layer_norm(x, block.ln_attn_gamma, block.ln_attn_beta)
    # Check that output has mean close to 0 and std close to 1 per feature
    mean = np.mean(x_normalized, axis=0)
    var = np.var(x_normalized, axis=0)
    assert np.allclose(mean, 0, atol=1e-5)
    assert np.allclose(var, 1, atol=1e-5)

def test_feed_forward_hidden_size():
    """FF hidden layer should expand then contract."""
    d_model = 64
    d_ff = 256
    block = TransformerBlock(d_model=d_model, num_heads=8, d_ff=d_ff)
    assert block.W_1.shape == (d_model, d_ff), "FF expand layer"
    assert block.W_2.shape == (d_ff, d_model), "FF contract layer"
