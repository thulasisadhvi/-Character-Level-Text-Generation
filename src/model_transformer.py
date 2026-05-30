import torch
import torch.nn as nn
import math

# 1. Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a matrix of shape (max_len, embedding_dim)
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, embedding_dim)
        
        # Register as a buffer so it's not a learnable parameter, but saves with the model
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encodings to the input embeddings
        x = x + self.pe[:, :x.size(1), :]
        return x

# 2. Multi-Head Self-Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert embedding_dim % n_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        
        # Linear layers for Query, Key, Value
        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)
        
        # Final output linear layer
        self.fc_out = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Calculate Q, K, V and split into multiple heads
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention: Q * K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply Causal Mask (prevent looking ahead)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-1e20'))
            
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Multiply attention weights by Values
        out = torch.matmul(attention_weights, V)
        
        # Recombine heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)
        return self.fc_out(out)

# 3. Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# 4. Encoder Block (Combines Attention and Feed-Forward with Normalization)
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, n_heads, hidden_dim):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embedding_dim, n_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        self.feed_forward = FeedForward(embedding_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
    def forward(self, x, mask=None):
        # Sub-layer 1: Multi-Head Attention + Add & Norm
        attention_out = self.attention(x, mask)
        x = self.norm1(x + attention_out)
        
        # Sub-layer 2: Feed Forward + Add & Norm
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x

# 5. Final Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, n_heads=4):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        # Stack multiple Encoder blocks
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(embedding_dim, n_heads, hidden_dim) for _ in range(n_layers)]
        )
        
        # Output layer maps back to vocabulary size
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        # hidden is ignored for transformer, included for API compatibility with LSTM in training loop
        batch_size, seq_len = x.shape
        device = x.device
        
        # Create lower-triangular causal mask
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).view(1, 1, seq_len, seq_len)
        
        # Pass through embedding and inject positional data
        out = self.embedding(x)
        out = self.pos_encoder(out)
        
        # Pass through all transformer blocks
        for layer in self.layers:
            out = layer(out, mask)
            
        # Reshape for final Linear layer (batch_size * seq_len, embedding_dim)
        out = out.contiguous().view(-1, out.size(-1))
        
        # Get raw logits
        logits = self.fc_out(out)
        
        return logits, None  # return None for hidden state to match LSTM return signature