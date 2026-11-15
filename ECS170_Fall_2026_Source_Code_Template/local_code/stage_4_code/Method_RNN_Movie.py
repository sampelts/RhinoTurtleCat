
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class Movie_RNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # Embedding layer converts word indices into vectors
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=128,
            padding_idx=0
        )

        # RNN layer processes the sequence of words
        self.rnn = nn.RNN(
            input_size=128,
            hidden_size=128,
            batch_first=True
        )

        # Fully connected layer for binary classification
        self.fc = nn.Linear(128, 1)

    def forward(self, x, lengths):

        # Convert word indices into embedding vectors
        x = self.embedding(x)

        # Pack sequences to ignore padding tokens
        x = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # Pass the packed sequences through the RNN
        output, hidden = self.rnn(x)

        # Get the final hidden state of the RNN
        x = hidden[-1]

        # Pass through fully connected layer
        x = self.fc(x)

        # Return one output per review
        return x.squeeze(1)
