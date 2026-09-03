"""
Positional encoding - add sequence order information to embeddings.
Without this, a Transformer treats "the cat sat on the mat" the same as "mat the on sat cat the".
Uses sine/cosine waves at different frequencies for each position and dimension.
"""

import numpy as np

class PositionalEncoding:
    """
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    pos: position in sequence (0, 1, 2, ...)
    i: dimension index (0, 1, 2, ..., d_model/2)
    d_model: total embedding dimension
    """
    
    def __init__(self, d_model, max_seq_len=512):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pe = self._compute_pe()
    
    def _compute_pe(self):
        """Precompute positional encodings for all positions."""
        pe = np.zeros((self.max_seq_len, self.d_model))
        
        positions = np.arange(self.max_seq_len)[:, np.newaxis]
        dimensions = np.arange(0, self.d_model, 2)[np.newaxis, :]
        
        # Compute angle rates
        angle_rates = 1 / np.power(10000, dimensions / np.float32(self.d_model))
        
        # Sine for even dimensions
        pe[:, 0::2] = np.sin(positions * angle_rates)
        
        # Cosine for odd dimensions
        pe[:, 1::2] = np.cos(positions * angle_rates)
        
        return pe
    
    def forward(self, embeddings):
        """
        Add positional encoding to embeddings.
        embeddings: (batch, seq_len, d_model) or (seq_len, d_model)
        Returns: embeddings + positional_encoding
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        
        if embeddings.ndim == 2:
            # (seq_len, d_model)
            seq_len = embeddings.shape[0]
            return embeddings + self.pe[:seq_len]
        elif embeddings.ndim == 3:
            # (batch, seq_len, d_model)
            seq_len = embeddings.shape[1]
            return embeddings + self.pe[:seq_len]
        else:
            raise ValueError(f"Expected 2D or 3D array, got {embeddings.ndim}D")
    
    def get_encoding(self, seq_len):
        """Get positional encoding for a specific sequence length."""
        return self.pe[:seq_len]
