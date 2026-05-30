# PyTorch Sequence Models from Scratch

This project implements two fundamental sequence models—a **Long Short-Term Memory (LSTM)** network and a **Mini-Transformer**—built entirely from scratch using PyTorch. The models are trained on a character-level text generation task using the works of William Shakespeare.

This project serves as a practical, hands-on exploration of sequence modeling, self-attention mechanisms, and generative AI concepts.

## 🧠 Core Architectures

1. **LSTM (Sequential):** Processes text character-by-character, updating a hidden state to retain context over time. Highly efficient for small-scale datasets and CPU training.
2. **Mini-Transformer (Parallel):** Uses Multi-Head Self-Attention to process sequence contexts simultaneously. Implements positional encoding, feed-forward networks, and causal masking to prevent forward-looking data leakage during training.

## 📁 Project Structure

```text
/
├── Dockerfile                  # Container definition for reproducible execution
├── docker-compose.yml          # Volume mounts and service configuration
├── .env.example                # Environment variables template
├── requirements.txt            # Python dependencies (PyTorch CPU, NumPy, Matplotlib)
├── input/
│   ├── shakespeare.txt         # Raw text dataset (Tiny Shakespeare)
│   └── mappings.json           # Character-to-integer vocabulary mappings
├── models/                     # Saved model weights (.pth)
├── results/                    # Generated text, loss curves, and evaluation reports
└── src/
    ├── prepare_data.py         # Data tokenization and batching logic
    ├── model_lstm.py           # LSTM architecture
    ├── model_transformer.py    # Transformer architecture
    ├── train.py                # Training loop (loss calculation, backprop, clipping)
    ├── generate.py             # Temperature-scaled text generation
    └── create_artifacts.py     # Evaluation graphing and JSON sample generation
