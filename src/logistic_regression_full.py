"""
Logistic Regression from scratch — Part 2: training loop with gradient descent.

Reuses the sigmoid and binary_cross_entropy from Part 1. The gradient update
is almost identical to linear regression, except:
  - We use sigmoid output, not raw prediction
  - Cost is binary cross-entropy, not MSE
  - Gradient derivation is slightly different (but looks remarkably similar)
"""

from __future__ import annotations
import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    eps = 1e-15
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    y_true = np.asarray(y_true, dtype=np.float64)
    costs = -(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
    return float(np.mean(costs))


class LogisticRegressionScratch:
    """Logistic regression trained with batch gradient descent."""

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, verbose: bool = False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        """Train with gradient descent on binary cross-entropy loss."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for i in range(self.n_iterations):
            z = X @ self.weights + self.bias
            y_pred = sigmoid(z)

            error = y_pred - y
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)

            if self.verbose and (i % max(1, self.n_iterations // 10) == 0):
                print(f"Iteration {i:5d} | Binary Cross-Entropy: {loss:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities."""
        if self.weights is None:
            raise RuntimeError("Model has not been fit yet.")
        X = np.asarray(X, dtype=np.float64)
        z = X @ self.weights + self.bias
        return sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return predicted class labels (0 or 1)."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy on test set."""
        y = np.asarray(y, dtype=np.float64)
        preds = self.predict(X)
        return float(np.mean(preds == y))
