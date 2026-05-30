import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # 1. Embedding Layer: Converts character integers to dense vectors.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. LSTM Layer: Processes the sequence of embeddings.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        
        # 3. Fully Connected Layer: Maps LSTM output to vocabulary space.
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden):
        # x shape: (batch_size, sequence_length)
        batch_size = x.size(0)
        
        # Pass input through the embedding layer
        embeds = self.embedding(x)
        
        # Pass embeddings through the LSTM layer
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        # Flatten the output from the LSTM so it can be passed to the Linear layer.
        # It reshapes from (batch_size, seq_len, hidden_dim) to (batch_size * seq_len, hidden_dim)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # Pass through the fully connected layer to get the final predictions (logits)
        out = self.fc(lstm_out)
        
        return out, hidden
        
    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state for the LSTM with zeros
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden