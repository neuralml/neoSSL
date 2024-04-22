import torch
from torch import nn

def get_activation(activation):
    if activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError

class DenoisingAE(nn.Module):
    def __init__(self, input_dim, latent_dim, activation='sigmoid', **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            get_activation(activation),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x