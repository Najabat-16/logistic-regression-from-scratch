"""Tests for LogisticRegressionScratch training."""
import sys, os
import numpy as np
import pytest
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.logistic_regression_full import LogisticRegressionScratch

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def generate_separable_data(n_samples=200, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_samples, 2))
    true_w = np.array([2.0, -1.5])
    z = X @ true_w + 0.5
    y = (sigmoid(z) > 0.5).astype(float)
    return X, y

def test_predict_proba_in_range():
    X, y = generate_separable_data(100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=500)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    assert np.all(proba >= 0) and np.all(proba <= 1)

def test_predict_binary_output():
    X, y = generate_separable_data(100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=500)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert np.all((preds == 0) | (preds == 1))

def test_converges_on_separable_data():
    X, y = generate_separable_data(200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    assert acc > 0.9

def test_loss_decreases():
    X, y = generate_separable_data(100)
    model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=500, verbose=False)
    model.fit(X, y)
    first_quarter = np.mean(model.loss_history[:125])
    last_quarter = np.mean(model.loss_history[-125:])
    assert last_quarter < first_quarter
