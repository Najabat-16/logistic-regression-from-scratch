# ML Fundamentals to RAG: From Scratch Portfolio

A complete learning progression from linear regression through retrieval-augmented generation, all implemented from scratch with NumPy. Every algorithm broken down to first principles — math visible, no black boxes.

## Modules

**Fundamentals**
- `linear_regression.py` — Batch gradient descent on MSE loss
- `logistic_regression_full.py` — Sigmoid + binary cross-entropy + backprop
- `neural_network.py` — 2-layer MLP with ReLU + backpropagation

**Representations & Context**
- `embeddings.py` — Skip-gram-like word embeddings via softmax prediction
- `attention.py` — Scaled dot-product attention (Transformer core)
- `tokenizer.py` — Text tokenization and vocabulary building

**RAG Pipeline**
- `rag_retriever.py` — Cosine similarity retrieval from vector database
- `rag_main.py` — End-to-end demo: tokenize -> embed -> retrieve

## Key Learning Progression

1. Linear Regression (MSE loss, gradient descent)
2. Logistic Regression (sigmoid, binary cross-entropy)
3. Neural Networks (ReLU, backpropagation chain rule)
4. Embeddings (learned dense vectors, softmax prediction)
5. Attention (query-key-value mechanism, causal masking)
6. Tokenizer (vocabulary, encoding/decoding)
7. RAG Retriever (cosine similarity, top-k retrieval)

## Test Coverage

38 unit tests across all modules, all passing:
- Linear regression: weight recovery, loss decrease, R2 score
- Logistic regression: sigmoid properties, convergence, accuracy
- Neural network: weight updates, loss decrease, generalization
- Embeddings: training convergence, similarity ranking
- Attention: output shape, weight probabilities, causal masking
- Tokenizer: vocab building, encode/decode, roundtrip
- RAG: document retrieval, top-k ranking, pipeline integration

## Philosophy

Most AI Engineer portfolios stack high-level libraries. This signals something different: you understand the math underneath.

A recruiter looking at this sees:
- Can implement backpropagation without a framework
- Knows why attention works (it's literally there in the code)
- Can derive gradients by hand
- Understands the whole pipeline from data -> embeddings -> retrieval

## Usage

pytest tests/ -v              # run all tests
python main.py                # neural network training
python rag_main.py            # RAG pipeline demo

## Next Steps

- Multi-head attention
- Positional encodings
- Full Transformer block
- LLM training loop
- Fine-tuning on domain data

## License

MIT
