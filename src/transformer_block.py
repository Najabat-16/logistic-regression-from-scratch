"""
Transformer block - complete layer with multi-head attention + feed-forward.
Includes layer normalization and residual connections (the actual magic).
"""

import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)

class TransformerBlock:
    """
    One transformer block: Multi-head attention -> Feed-forward network
    With residual connections and layer normalization (post-norm style)
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.d_k = d_model // num_heads
        
        # Multi-head attention
        self.W_Q = np.random.randn(d_model, d_model) * 0.01
        self.W_K = np.random.randn(d_model, d_model) * 0.01
        self.W_V = np.random.randn(d_model, d_model) * 0.01
        self.W_O = np.random.randn(d_model, d_model) * 0.01
        
        # Feed-forward network
        self.W_1 = np.random.randn(d_model, d_ff) * 0.01
        self.b_1 = np.zeros((1, d_ff))
        self.W_2 = np.random.randn(d_ff, d_model) * 0.01
        self.b_2 = np.zeros((1, d_model))
        
        # Layer normalization
        self.ln_attn_gamma = np.ones((1, d_model))
        self.ln_attn_beta = np.zeros((1, d_model))
        self.ln_ff_gamma = np.ones((1, d_model))
        self.ln_ff_beta = np.zeros((1, d_model))
        
        self.attention_weights = None
    
    def layer_norm(self, x, gamma, beta, eps=1e-6):
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def combine_heads(self, x):
        batch_size, num_heads, seq_len, d_k = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, num_heads * d_k)
    
    def forward(self, x, mask=None):
        """
        Forward pass through transformer block.
        x: (batch, seq_len, d_model)
        """
        batch_size = x.shape[0]
        
        # Multi-head attention
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V
        
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores + mask[np.newaxis, np.newaxis, :, :]
        
        attention_weights = softmax(scores, axis=-1)
        self.attention_weights = attention_weights
        
        context = attention_weights @ V
        context = self.combine_heads(context)
        attn_output = context @ self.W_O
        
        # Residual connection + layer norm
        x_norm = self.layer_norm(x + attn_output, self.ln_attn_gamma, self.ln_attn_beta)
        
        # Feed-forward network
        ff_hidden = self.relu(x_norm @ self.W_1 + self.b_1)
        ff_output = ff_hidden @ self.W_2 + self.b_2
        
        # Residual connection + layer norm
        output = self.layer_norm(x_norm + ff_output, self.ln_ff_gamma, self.ln_ff_beta)
        
        return output
