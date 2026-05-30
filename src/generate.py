import argparse
import torch
import torch.nn.functional as F
import json
import os

from model_lstm import LSTMModel
from model_transformer import TransformerModel

def generate_text(model, start_seq, char_to_int, int_to_char, vocab_size, device, max_length=200, temperature=1.0):
    model.eval() # Set model to evaluation mode
    
    # Convert starting string to integer tensor
    chars = [char_to_int[c] for c in start_seq]
    input_seq = torch.tensor(chars, dtype=torch.long).unsqueeze(0).to(device)

    generated_text = start_seq

    # Initialize hidden state for LSTM
    if isinstance(model, LSTMModel):
        hidden = model.init_hidden(1) # Batch size of 1 for generation
        hidden = tuple([each.to(device) for each in hidden])

    with torch.no_grad(): # No need to calculate gradients for generation
        for _ in range(max_length):
            
            if isinstance(model, LSTMModel):
                # For LSTM, we feed one character at a time and pass the hidden state forward
                out, hidden = model(input_seq[:, -1:], hidden)
                logits = out[-1, :] # Get the last set of logits
            else:
                # For Transformer, we feed the whole sequence context
                out, _ = model(input_seq)
                logits = out[-1, :] # Get the very last prediction

            # Apply temperature scaling
            logits = logits / temperature
            probs = F.softmax(logits, dim=0)
            
            # Sample from the probability distribution
            next_char_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = int_to_char[next_char_idx]
            
            generated_text += next_char
            
            # Append the new character to the sequence for the next iteration
            next_char_tensor = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)
            input_seq = torch.cat((input_seq, next_char_tensor), dim=1)

    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Generate text using trained model.")
    parser.add_argument('--model', type=str, required=True, choices=['lstm', 'transformer'], help='Model type')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model weights')
    parser.add_argument('--seed_text', type=str, default="O Romeo, Romeo!", help='Text to start generation')
    parser.add_argument('--temperature', type=float, default=1.0, help='Creativity scaling factor')
    parser.add_argument('--length', type=int, default=300, help='Number of characters to generate')
    args = parser.parse_args()

    device = torch.device("cpu")

    # Load mappings
    with open('input/mappings.json', 'r') as f:
        mappings = json.load(f)
        
    char_to_int = mappings['char_to_int']
    # JSON saves dictionary keys as strings, so we convert them back to integers here
    int_to_char = {int(k): v for k, v in mappings['int_to_char'].items()}
    vocab_size = mappings['vocab_size']

    # Hyperparameters (Must match training exactly)
    embedding_dim = 64
    hidden_dim = 128
    n_layers = 2

    # Instantiate the correct model
    if args.model == 'lstm':
        model = LSTMModel(vocab_size, embedding_dim, hidden_dim, n_layers).to(device)
    else:
        model = TransformerModel(vocab_size, embedding_dim, hidden_dim, n_layers, n_heads=4).to(device)

    # Load the trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    print(f"\n--- Generating text with {args.model.upper()} (Temp: {args.temperature}) ---")
    generated = generate_text(model, args.seed_text, char_to_int, int_to_char, vocab_size, device, args.length, args.temperature)
    
    print("\n" + generated)
    print("\n---------------------------------------------------------")

if __name__ == '__main__':
    main()