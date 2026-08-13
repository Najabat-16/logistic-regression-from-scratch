# Neural Network from Scratch

A 2-layer fully connected neural network with ReLU hidden activation and sigmoid output, trained via backpropagation on binary classification tasks.

## Architecture

Input Layer (15) -> Hidden Layer (32, ReLU) -> Output Layer (1, sigmoid)

## Forward Pass

z1 = X @ W1 + b1
a1 = ReLU(z1)
z2 = a1 @ W2 + b2
a2 = sigmoid(z2)

## Backpropagation

dz2 = a2 - y
dW2 = (1/n) * a1^T @ dz2
da1 = dz2 @ W2^T
dz1 = da1 * ReLU'(z1)
dW1 = (1/n) * X^T @ dz1

## Results

Train Accuracy: 96.56%
Test Accuracy: 82.50%
Final Loss: 0.1479

## Key Points

1. Hidden layer adds nonlinearity
2. ReLU derivative is (z > 0)
3. Chain rule through both layers
4. More parameters than logistic regression

## Usage

python main.py       # train and plot results
pytest tests/ -v     # run all tests
