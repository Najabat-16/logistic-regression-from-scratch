"""Tests for NeuralNetworkScratch backpropagation and training."""
import sys, os
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.neural_network import NeuralNetworkScratch

def test_network_weight_update():
    """Weights should change after backward pass."""
    X = np.random.randn(10, 5)
    y = np.random.randint(0, 2, (10, 1))
    model = NeuralNetworkScratch(input_size=5, hidden_size=8, learning_rate=0.1, n_iterations=1)
    W1_before = model.W1.copy()
    model.fit(X, y)
    assert not np.allclose(W1_before, model.W1), "Weights should update"

def test_forward_output_shape():
    """Output should have correct shape."""
    X = np.random.randn(20, 5)
    model = NeuralNetworkScratch(input_size=5, hidden_size=8)
    output = model.forward(X)
    assert output.shape == (20, 1)

def test_loss_decreases():
    """Loss should trend downward over iterations."""
    X, y = make_classification(n_samples=150, n_features=8, n_informative=6, random_state=42)
    model = NeuralNetworkScratch(input_size=8, hidden_size=16, learning_rate=0.1, n_iterations=200)
    model.fit(X, y)
    first_20 = np.mean(model.loss_history[:20])
    last_20 = np.mean(model.loss_history[-20:])
    assert last_20 < first_20, "Loss should decrease"

def test_predict_binary_output():
    """Predictions should be 0 or 1."""
    X, y = make_classification(n_samples=100, n_features=6, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = NeuralNetworkScratch(input_size=6, hidden_size=8, n_iterations=500)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert np.all((preds == 0) | (preds == 1))

def test_network_learns():
    """Network should achieve reasonable accuracy on classification task."""
    X, y = make_classification(n_samples=300, n_features=10, n_informative=8, n_redundant=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = NeuralNetworkScratch(input_size=10, hidden_size=16, learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    assert acc > 0.75, f"Expected accuracy >0.75, got {acc:.4f}"
