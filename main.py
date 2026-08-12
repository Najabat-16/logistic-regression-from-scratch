"""
Full neural network pipeline: train, evaluate, plot loss curve.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.neural_network import NeuralNetworkScratch

# Generate data
X, y = make_classification(n_samples=400, n_features=15, n_informative=10, n_redundant=3, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = NeuralNetworkScratch(input_size=15, hidden_size=32, learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train, verbose=True)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"\nTrain Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(model.loss_history, color="#4C72B0")
plt.title("Training Loss (Binary Cross-Entropy)")
plt.xlabel("Iteration")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.text(0.5, 0.5, f"Train Acc: {train_acc:.4f}\nTest Acc: {test_acc:.4f}", 
         ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
plt.axis('off')
plt.tight_layout()
plt.savefig("outputs/nn_training.png", dpi=150)
print("Saved plot to outputs/nn_training.png")
