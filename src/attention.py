"""
Scaled dot-product attention - core mechanism in Transformers.
Computes attention weights between queries and keys, uses them to weight values.
"""

import numpy as np

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)

class ScaledDotProductAttention:
    """
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Q: queries (seq_len, d_k)
    K: keys (seq_len, d_k)
    V: values (seq_len, d_v)
    """
    
    def __init__(self, d_k):
        self.d_k = d_k
        self.attention_weights = None
    
    def forward(self, Q, K, V, mask=None):
        """
        Q: (batch, seq_len, d_k)
        K: (batch, seq_len, d_k)
        V: (batch, seq_len, d_v)
        mask: optional mask to prevent attending to future tokens
        """
        Q = np.asarray(Q, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        
        # Compute attention scores
        scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(self.d_k)  # (batch, seq_len, seq_len)
        
        # Apply mask if provided (for causal attention)
        if mask is not None:
            scores = scores + mask
        
        # Softmax over last dimension
        self.attention_weights = softmax(scores, axis=-1)  # (batch, seq_len, seq_len)
        
        # Apply attention to values
        output = self.attention_weights @ V  # (batch, seq_len, d_v)
        
        return output
    
    def backward(self, doutput, Q, K, V, mask=None):
        """
        Backpropagation through attention.
        Returns: dQ, dK, dV
        """
        # Gradient w.r.t. values
        dV = self.attention_weights.transpose(0, 2, 1) @ doutput
        
        # Gradient w.r.t. attention weights
        d_attention = doutput @ V.transpose(0, 2, 1)
        
        # Gradient of softmax: diag(p) - p @ p^T
        batch_size, seq_len, _ = self.attention_weights.shape
        d_scores = np.zeros_like(d_attention)
        for b in range(batch_size):
            p = self.attention_weights[b]  # (seq_len, seq_len)
            jacobian = p[:, :, np.newaxis] * (np.eye(seq_len) - p[np.newaxis, :, :])
            d_scores[b] = (jacobian * d_attention[b]).sum(axis=1)
        
        # Scale gradient
        d_scores = d_scores / np.sqrt(self.d_k)
        
        # Gradient w.r.t. Q and K
        dQ = d_scores @ K
        dK = d_scores.transpose(0, 2, 1) @ Q
        
        return dQ, dK, dV

def causal_mask(seq_len):
    """Create causal mask for autoregressive models."""
    mask = np.tril(np.ones((seq_len, seq_len)))
    return np.where(mask == 0, -1e10, 0)
