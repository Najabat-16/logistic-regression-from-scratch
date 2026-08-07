"""Unit tests for the Part 1 math core: sigmoid + binary cross-entropy."""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.logistic_regression import sigmoid, binary_cross_entropy, predict_proba


def test_sigmoid_at_zero_is_half():
    assert np.isclose(sigmoid(0), 0.5)


def test_sigmoid_output_range():
    z = np.array([-1000, -10, 0, 10, 1000])
    out = sigmoid(z)
    assert np.all(out >= 0) and np.all(out <= 1)


def test_sigmoid_monotonic_increasing():
    z = np.linspace(-10, 10, 50)
    out = sigmoid(z)
    assert np.all(np.diff(out) > 0)


def test_bce_perfect_prediction_near_zero():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.999, 0.001, 0.999, 0.001])
    cost = binary_cross_entropy(y_true, y_pred)
    assert cost < 0.01


def test_bce_confident_wrong_prediction_is_high():
    y_true = np.array([1, 0])
    y_pred = np.array([0.001, 0.999])  # confidently wrong both times
    cost = binary_cross_entropy(y_true, y_pred)
    assert cost > 5.0


def test_predict_proba_shape_and_range():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    weights = np.array([0.5, -0.3])
    bias = 0.1
    probs = predict_proba(X, weights, bias)
    assert probs.shape == (3,)
    assert np.all(probs >= 0) and np.all(probs <= 1)
