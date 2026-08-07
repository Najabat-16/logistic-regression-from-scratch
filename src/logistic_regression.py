"""
Logistic Regression from scratch — Part 1: core math.

Linear regression predicts a continuous number. Logistic regression predicts
a PROBABILITY (0 to 1) by squashing a linear combination through the sigmoid
function. This file implements just that squashing + the cost function that
scores how good a set of weights is. Gradient descent training (Part 2) will
reuse the update-rule logic from the linear regression project almost
unchanged — the only thing that's genuinely new here is these two pieces.

    z = X @ w + b                      (same linear step as before)
    y_hat = sigmoid(z) = 1 / (1 + e^-z)  (new: squash into a 0-1 probability)
    Loss = binary cross-entropy, not MSE (new: MSE doesn't fit a probability output)
"""

from __future__ import annotations
import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Squash any real number into the range (0, 1).

    sigmoid(0)   = 0.5   (maximum uncertainty)
    sigmoid(+inf) -> 1   (confident positive class)
    sigmoid(-inf) -> 0   (confident negative class)

    Clipped internally to avoid overflow warnings on extreme inputs.
    """
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Cost function for classification (replaces MSE from linear regression).

    For each sample:
        cost = -[ y*log(y_hat) + (1-y)*log(1-y_hat) ]

    Intuition: if the true label is 1 and the model predicted 0.99, cost is
    tiny. If the true label is 1 and the model predicted 0.01, cost is huge.
    It punishes confident WRONG predictions much harder than MSE would.

    Clipped to avoid log(0).
    """
    eps = 1e-15
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    y_true = np.asarray(y_true, dtype=np.float64)

    costs = -(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
    return float(np.mean(costs))


def predict_proba(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """Compute predicted probabilities for each sample (the 'forward pass')."""
    X = np.asarray(X, dtype=np.float64)
    z = X @ weights + bias
    return sigmoid(z)
