import json
import matplotlib.pyplot as plt
import torch
import os
from model_lstm import LSTMModel
from model_transformer import TransformerModel
from generate import generate_text

def plot_losses():
    print("Plotting loss curves...")
    with open('results/lstm_loss_history.json', 'r') as f:
        lstm_loss = json.load(f)
    
    try:
        with open('results/transformer_loss_history.json', 'r') as f:
            tf_loss = json.load(f)
    except FileNotFoundError:
        print("Transformer loss file not found! Did it finish training?")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(lstm_loss, label='LSTM Loss', color='blue')
    plt.plot(tf_loss, label='Transformer Loss', color='orange')
    plt.title('Training Loss: LSTM vs Mini-Transformer')
    plt.xlabel('Training Steps (x100 batches)')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/loss_curves.png')
    print("Saved plot to results/loss_curves.png")

def generate_json_samples():
    print("Generating samples for JSON output. This might take a minute...")
    device = torch.device("cpu")
    
    with open('input/mappings.json', 'r') as f:
        mappings = json.load(f)
    char_to_int = mappings['char_to_int']
    int_to_char = {int(k): v for k, v in mappings['int_to_char'].items()}
    vocab_size = mappings['vocab_size']

    # Initialize models
    lstm = LSTMModel(vocab_size, 64, 128, 2).to(device)
    lstm.load_state_dict(torch.load('models/lstm_model.pth', map_location=device, weights_only=True))
    
    tf = TransformerModel(vocab_size, 64, 128, 2, n_heads=4).to(device)
    tf.load_state_dict(torch.load('models/transformer_model.pth', map_location=device, weights_only=True))

    models = {'lstm': lstm, 'transformer': tf}
    temperatures = [0.5, 1.0, 1.5]
    seed = "KING RICHARD:"
    
    results = {"lstm": {}, "transformer": {}}

    for name, model in models.items():
        for temp in temperatures:
            key = f"temperature_{temp}"
            print(f"Generating {name} at {key}...")
            # Generate 2 samples per temperature
            sample1 = generate_text(model, seed, char_to_int, int_to_char, vocab_size, device, 150, temp)
            sample2 = generate_text(model, seed, char_to_int, int_to_char, vocab_size, device, 150, temp)
            results[name][key] = [sample1, sample2]

    with open('results/generated_samples.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Saved JSON samples to results/generated_samples.json")

if __name__ == '__main__':
    plot_losses()
    generate_json_samples()