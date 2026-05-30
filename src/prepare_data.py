import os
import torch
import json
import numpy as np

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def create_vocab_and_mappings(text):
    # Identify all unique characters
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # Build dictionaries
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    
    return chars, vocab_size, char_to_int, int_to_char

def encode_text(text, char_to_int):
    return np.array([char_to_int[c] for c in text])

def get_batches(encoded_text, batch_size, seq_length):
    """
    Yields batches of input-target pairs.
    """
    total_batch_size = batch_size * seq_length
    n_batches = len(encoded_text) // total_batch_size
    
    # Keep only enough characters to make full batches
    encoded_text = encoded_text[:n_batches * total_batch_size]
    
    # Reshape into (batch_size, -1)
    encoded_text = encoded_text.reshape((batch_size, -1))
    
    for n in range(0, encoded_text.shape[1], seq_length):
        # The inputs
        x = encoded_text[:, n:n+seq_length]
        # The targets (shifted by 1)
        y = np.zeros_like(x)
        try:
            y[:, :-1] = x[:, 1:]
            y[:, -1] = encoded_text[:, n+seq_length]
        except IndexError:
            # For the very last batch, pad with the first character to avoid error
            y[:, :-1] = x[:, 1:]
            y[:, -1] = encoded_text[:, 0]
            
        yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def main():
    # Define paths
    input_file = 'input/shakespeare.txt'
    mapping_file = 'input/mappings.json'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please place your text dataset there.")
        return

    print("Loading text...")
    text = load_text(input_file)
    print(f"Corpus length: {len(text)} characters.")
    
    print("Creating vocabulary...")
    chars, vocab_size, char_to_int, int_to_char = create_vocab_and_mappings(text)
    print(f"Vocabulary size: {vocab_size} unique characters.")
    
    # Save mappings for the generation script later
    with open(mapping_file, 'w') as f:
        json.dump({
            'char_to_int': char_to_int,
            'int_to_char': int_to_char,
            'vocab_size': vocab_size
        }, f)
    print(f"Mappings saved to {mapping_file}")
    
    print("Encoding dataset...")
    encoded_text = encode_text(text, char_to_int)
    print("Data preparation complete. Ready for modeling!")

if __name__ == '__main__':
    main()