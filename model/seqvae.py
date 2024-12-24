import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder

class SeqVAE(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, latent_dim=16, seq_length=50):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, drop_prob=0.3)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, seq_length, drop_prob=0.3)
        
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        out = self.decoder(z, seq_length=x.shape[1])
        return out, mean, log_var
    
    def sample(self, num_samples, seq_length=None, device='cpu'):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        with torch.no_grad():
            generated = self.decoder(z, seq_length=seq_length)
        return generated