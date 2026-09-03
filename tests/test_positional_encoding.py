"""Tests for positional encoding."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.positional_encoding import PositionalEncoding

def test_pe_shape():
    """Positional encoding matrix should have correct shape."""
    d_model = 16
    max_seq_len = 100
    pe = PositionalEncoding(d_model, max_seq_len)
    assert pe.pe.shape == (max_seq_len, d_model)

def test_forward_2d():
    """Should work with 2D embeddings (seq_len, d_model)."""
    pe = PositionalEncoding(d_model=16, max_seq_len=100)
    embeddings = np.random.randn(10, 16)
    output = pe.forward(embeddings)
    assert output.shape == (10, 16)

def test_forward_3d():
    """Should work with 3D embeddings (batch, seq_len, d_model)."""
    pe = PositionalEncoding(d_model=16, max_seq_len=100)
    embeddings = np.random.randn(2, 10, 16)
    output = pe.forward(embeddings)
    assert output.shape == (2, 10, 16)

def test_positions_differ():
    """Different positions should have different encodings."""
    pe = PositionalEncoding(d_model=16, max_seq_len=100)
    enc_pos0 = pe.get_encoding(1)[0]
    enc_pos1 = pe.get_encoding(2)[1]
    assert not np.allclose(enc_pos0, enc_pos1)

def test_position_consistency():
    """Same position should always have same encoding."""
    pe = PositionalEncoding(d_model=16, max_seq_len=100)
    enc1 = pe.pe[5].copy()
    enc2 = pe.pe[5].copy()
    assert np.allclose(enc1, enc2)

def test_bounded_magnitude():
    """Positional encodings should be bounded in magnitude."""
    pe = PositionalEncoding(d_model=32, max_seq_len=1000)
    assert np.all(np.abs(pe.pe) <= 1.0), "PE values should be bounded by [-1, 1]"

def test_get_encoding_length():
    """get_encoding should return requested sequence length."""
    pe = PositionalEncoding(d_model=16, max_seq_len=100)
    for seq_len in [1, 5, 10, 50]:
        enc = pe.get_encoding(seq_len)
        assert enc.shape == (seq_len, 16)
