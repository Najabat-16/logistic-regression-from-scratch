"""
Simple tokenizer - convert text to token indices and back.
Builds vocabulary from text corpus.
"""

import numpy as np
from collections import Counter

class SimpleTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab = []
    
    def build_vocab(self, texts, min_freq=1):
        counter = Counter()
        for text in texts:
            words = text.lower().split()
            counter.update(words)
        
        most_common = counter.most_common(self.vocab_size - 2)
        self.word_to_idx['<PAD>'] = 0
        self.word_to_idx['<UNK>'] = 1
        
        idx = 2
        for word, freq in most_common:
            if freq >= min_freq:
                self.word_to_idx[word] = idx
                idx += 1
        
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        self.vocab = list(self.word_to_idx.keys())
    
    def encode(self, text, max_len=None):
        words = text.lower().split()
        tokens = [self.word_to_idx.get(w, self.word_to_idx['<UNK>']) for w in words]
        
        if max_len:
            if len(tokens) < max_len:
                tokens = tokens + [self.word_to_idx['<PAD>']] * (max_len - len(tokens))
            else:
                tokens = tokens[:max_len]
        
        return np.array(tokens, dtype=np.int64)
    
    def decode(self, token_indices):
        words = [self.idx_to_word.get(int(idx), '<UNK>') for idx in token_indices]
        words = [w for w in words if w not in ['<PAD>', '<UNK>']]
        return ' '.join(words)
    
    def __len__(self):
        return len(self.word_to_idx)
