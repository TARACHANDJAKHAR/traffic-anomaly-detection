"""
Model class definitions.
Must match the architecture used during training in the notebook.
"""
import numpy as np
import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc   = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers, dropout, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.fc   = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.out  = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = self.fc(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        o, _ = self.lstm(h)
        return self.out(o)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, dropout, seq_len):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim,
                                   num_layers, dropout, seq_len)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x):
        with torch.no_grad():
            return ((x - self(x)) ** 2).mean(dim=(1, 2))


class FeatureNormalizer:
    def __init__(self):
        self.mean = self.std = None

    def fit(self, X):
        flat = X.reshape(-1, X.shape[-1])
        self.mean = flat.mean(0)
        self.std  = flat.std(0) + 1e-8

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path):
        d = np.load(path)
        self.mean = d["mean"]
        self.std  = d["std"]
