"""Tests for SimpleTokenizer."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.tokenizer import SimpleTokenizer

def test_build_vocab():
    """Should build vocabulary from texts."""
    texts = ["hello world", "hello there", "world of nlp"]
    tok = SimpleTokenizer(vocab_size=100)
    tok.build_vocab(texts)
    assert len(tok) > 0
    assert '<PAD>' in tok.word_to_idx
    assert '<UNK>' in tok.word_to_idx

def test_encode_basic():
    """Should encode text to token indices."""
    texts = ["hello world", "hello there"]
    tok = SimpleTokenizer(vocab_size=100)
    tok.build_vocab(texts)
    encoded = tok.encode("hello world")
    assert isinstance(encoded, np.ndarray)
    assert encoded.dtype == np.int64

def test_encode_with_max_len():
    """Should pad or truncate to max_len."""
    texts = ["hello world", "hello there", "foo bar"]
    tok = SimpleTokenizer(vocab_size=100)
    tok.build_vocab(texts)
    encoded = tok.encode("hello world", max_len=5)
    assert encoded.shape == (5,)

def test_encode_unknown_words():
    """Unknown words should map to <UNK>."""
    texts = ["hello world"]
    tok = SimpleTokenizer(vocab_size=100)
    tok.build_vocab(texts)
    encoded = tok.encode("xyzzy unknown")
    unk_idx = tok.word_to_idx['<UNK>']
    assert unk_idx in encoded

def test_decode():
    """Should decode token indices back to text."""
    texts = ["hello world", "hello there"]
    tok = SimpleTokenizer(vocab_size=100)
    tok.build_vocab(texts)
    encoded = tok.encode("hello world")
    decoded = tok.decode(encoded)
    assert "hello" in decoded
    assert "world" in decoded

def test_roundtrip():
    """Text -> encode -> decode should preserve words."""
    texts = ["machine learning", "deep learning", "nlp models"]
    tok = SimpleTokenizer(vocab_size=100)
    tok.build_vocab(texts)
    original = "machine learning"
    encoded = tok.encode(original)
    decoded = tok.decode(encoded)
    for word in original.split():
        assert word in decoded, f"Word '{word}' lost in roundtrip"
