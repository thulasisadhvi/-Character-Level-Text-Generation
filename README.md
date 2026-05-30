
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

```

## 🚀 Getting Started

This project is fully containerized to ensure it runs consistently on any machine without complex dependency management. It is optimized for CPU execution.

### Prerequisites

* [Docker](https://docs.docker.com/get-docker/)
* [Docker Compose](https://docs.docker.com/compose/install/)

### Setup

1. **Clone the repository and enter the directory:**
```bash
git clone https://github.com/thulasisadhvi/-Character-Level-Text-Generation
cd pytorch

```


2. **Download the Dataset:**
Ensure you have a text file named `shakespeare.txt` inside the `input/` directory.
```bash
curl -o input/shakespeare.txt [https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

```


3. **Build the Docker Environment:**
```bash
docker-compose build

```



## 🛠️ Usage Pipeline

Run the following commands sequentially to train and evaluate the models.

### 1. Data Preparation

Reads the dataset, generates the vocabulary, creates integer mappings, and prepares the batches.

```bash
docker-compose run --rm app python src/prepare_data.py

```

### 2. Training

Train both models. You can adjust the `--epochs` parameter (default is 10).

```bash
# Train the LSTM
docker-compose run --rm app python src/train.py --model lstm --epochs 5

# Train the Transformer
docker-compose run --rm app python src/train.py --model transformer --epochs 5

```

### 3. Text Generation (Manual Testing)

Generate text interactively. The `--temperature` flag controls creativity (e.g., `0.5` is repetitive/safe, `1.0` is balanced, `1.5` is chaotic).

```bash
docker-compose run --rm app python src/generate.py \
  --model lstm \
  --model_path models/lstm_model.pth \
  --seed_text "To be or not to be" \
  --temperature 1.0

```

### 4. Create Artifacts

Automatically generates the comparative loss curves and structured JSON files containing generated text samples at various temperatures.

```bash
docker-compose run --rm app python src/create_artifacts.py

```

## 📊 Results & Evaluation

After running the pipeline, check the `results/` folder:

* **`loss_curves.png`**: A visualization comparing the training stability and convergence of the LSTM vs. the Transformer.
* **`generated_samples.json`**: Output samples demonstrating the models' language acquisition and the effect of temperature scaling.
* **`comparison_report.md`**: An analysis of model perplexity and qualitative text formatting.
