"""
Word embeddings from scratch - learning dense vector representations.
Uses a simple skip-gram-like approach: predict context from center word.
"""

import numpy as np

class SimpleEmbedding:
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.01, n_iterations=1000):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.loss_history = []
        
        # Embedding matrix: vocab_size x embedding_dim
        self.W = np.random.randn(vocab_size, embedding_dim) * 0.01
        # Context prediction matrix: embedding_dim x vocab_size
        self.V = np.random.randn(embedding_dim, vocab_size) * 0.01

    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, center_idx, context_idx):
        # center_idx: (batch,) indices of center words
        # context_idx: (batch,) indices of context words
        
        # Get embeddings for center words
        center_embed = self.W[center_idx]  # (batch, embedding_dim)
        
        # Predict context logits
        logits = center_embed @ self.V  # (batch, vocab_size)
        probs = self.softmax(logits)    # (batch, vocab_size)
        
        # Cross-entropy loss
        batch_size = len(center_idx)
        correct_probs = probs[np.arange(batch_size), context_idx]
        loss = -np.mean(np.log(correct_probs + 1e-15))
        
        return probs, loss

    def backward(self, center_idx, context_idx, probs):
        batch_size = len(center_idx)
        
        # Gradient of softmax cross-entropy
        dlogits = probs.copy()
        dlogits[np.arange(batch_size), context_idx] -= 1
        dlogits /= batch_size
        
        # Gradient w.r.t. V
        center_embed = self.W[center_idx]
        dV = center_embed.T @ dlogits
        
        # Gradient w.r.t. center embeddings
        dW_center = dlogits @ self.V.T
        
        # Update
        self.V -= self.learning_rate * dV
        self.W[center_idx] -= self.learning_rate * dW_center

    def fit(self, center_indices, context_indices):
        """
        center_indices: array of center word indices
        context_indices: array of context word indices (same length)
        """
        n_samples = len(center_indices)
        batch_size = 32
        
        for iteration in range(self.n_iterations):
            perm = np.random.permutation(n_samples)
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                batch_idx = perm[i:i+batch_size]
                center_batch = center_indices[batch_idx]
                context_batch = context_indices[batch_idx]
                
                probs, loss = self.forward(center_batch, context_batch)
                epoch_loss += loss
                n_batches += 1
                self.backward(center_batch, context_batch, probs)
            
            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)
        
        return self

    def get_embedding(self, word_idx):
        """Return embedding vector for a word."""
        return self.W[word_idx]

    def most_similar(self, word_idx, topk=5):
        """Find most similar words using cosine similarity."""
        embedding = self.W[word_idx]
        # Cosine similarity
        similarity = (self.W @ embedding) / (np.linalg.norm(self.W, axis=1) * np.linalg.norm(embedding) + 1e-15)
        similar_idx = np.argsort(similarity)[-topk-1:-1][::-1]
        return similar_idx, similarity[similar_idx]
