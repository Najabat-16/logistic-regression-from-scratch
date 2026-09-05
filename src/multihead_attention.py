"""
Multi-head attention - run scaled dot-product attention in parallel.
Allows the model to attend to different parts of the input simultaneously.
Each head learns a different representation subspace.
"""

import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)

class MultiHeadAttention:
    """
    Multi-head attention with num_heads parallel attention mechanisms.
    d_model = num_heads * d_k
    """
    
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Projection matrices for each head
        self.W_Q = np.random.randn(d_model, d_model) * 0.01
        self.W_K = np.random.randn(d_model, d_model) * 0.01
        self.W_V = np.random.randn(d_model, d_model) * 0.01
        self.W_O = np.random.randn(d_model, d_model) * 0.01
        
        self.attention_weights = None
    
    def split_heads(self, x):
        """
        Split last dimension into (num_heads, d_k).
        Input: (batch, seq_len, d_model)
        Output: (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def combine_heads(self, x):
        """
        Reverse of split_heads.
        Input: (batch, num_heads, seq_len, d_k)
        Output: (batch, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, num_heads * d_k)
    
    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: (batch, seq_len, d_model)
        mask: optional causal mask
        Returns: (batch, seq_len, d_model)
        """
        batch_size = Q.shape[0]
        
        # Project and split into heads
        Q = (Q @ self.W_Q)
        K = (K @ self.W_K)
        V = (V @ self.W_V)
        
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Scaled dot-product attention per head
        scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores + mask[np.newaxis, np.newaxis, :, :]
        
        attention_weights = softmax(scores, axis=-1)
        self.attention_weights = attention_weights
        
        context = attention_weights @ V  # (batch, num_heads, seq_len, d_k)
        
        # Combine heads
        context = self.combine_heads(context)  # (batch, seq_len, d_model)
        
        # Final output projection
        output = context @ self.W_O
        
        return output
