import argparse
import torch
import torch.nn as nn
import json
import os

# Import our custom modules
from model_lstm import LSTMModel
from model_transformer import TransformerModel
from prepare_data import load_text, encode_text, get_batches

def main():
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="Train a character-level text generation model.")
    parser.add_argument('--model', type=str, required=True, choices=['lstm', 'transformer'], help='Model type to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seq_length', type=int, default=100, help='Sequence length')
    args = parser.parse_args()

    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 2. Load Data and Mappings
    print("Loading data and mappings...")
    text = load_text('input/shakespeare.txt')
    with open('input/mappings.json', 'r') as f:
        mappings = json.load(f)
    
    char_to_int = mappings['char_to_int']
    vocab_size = mappings['vocab_size']
    encoded_text = encode_text(text, char_to_int)

    # Model Hyperparameters (Kept small so CPU training is reasonable)
    embedding_dim = 64
    hidden_dim = 128
    n_layers = 2

    # 3. Instantiate Model, Loss, and Optimizer
    device = torch.device("cpu") # Forcing CPU as per requirements
    print(f"Initializing {args.model.upper()} model...")
    
    if args.model == 'lstm':
        model = LSTMModel(vocab_size, embedding_dim, hidden_dim, n_layers).to(device)
    else:
        model = TransformerModel(vocab_size, embedding_dim, hidden_dim, n_layers, n_heads=4).to(device)

    # CrossEntropyLoss expects raw logits and integer targets
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 4. The Training Loop
    loss_history = []
    print(f"Starting training for {args.epochs} epochs...")
    model.train()
    
    for epoch in range(args.epochs):
        # Generate batches for this epoch
        batches = get_batches(encoded_text, args.batch_size, args.seq_length)
        
        # Initialize LSTM hidden state once per epoch
        if args.model == 'lstm':
            hidden = model.init_hidden(args.batch_size)

        for batch_idx, (x, y) in enumerate(batches):
            x, y = x.to(device), y.to(device)

            # a. Zero the gradients
            optimizer.zero_grad()

            # b & c. Forward pass
            if args.model == 'lstm':
                # Detach hidden state to prevent backpropagating through entire history
                hidden = tuple([each.data for each in hidden])
                out, hidden = model(x, hidden)
            else:
                # Transformer doesn't use recurring hidden states
                out, _ = model(x)

            # d. Calculate loss (Flatten targets and outputs for CrossEntropy)
            loss = criterion(out, y.contiguous().view(-1))
            
            # e. Backpropagation
            loss.backward()

            # f. Gradient Clipping (Prevents Exploding Gradients / NaN loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # g. Update the weights
            optimizer.step()

            # Periodically print progress
            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch+1}/{args.epochs} | Batch: {batch_idx} | Loss: {loss.item():.4f}")
                loss_history.append(loss.item())

    # 5. Save Model and Results
    print("Training complete! Saving artifacts...")
    save_path = f'models/{args.model}_model.pth'
    torch.save(model.state_dict(), save_path)
    
    loss_file = f'results/{args.model}_loss_history.json'
    with open(loss_file, 'w') as f:
        json.dump(loss_history, f)
        
    print(f"Model saved to: {save_path}")
    print(f"Loss history saved to: {loss_file}")

if __name__ == '__main__':
    main()