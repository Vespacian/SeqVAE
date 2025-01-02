import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, latent_dim=16, drop_prob=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=drop_prob, num_layers=3)
        self.mean_fc = nn.Linear(hidden_dim, latent_dim)
        self.log_var_fc = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        # x: (B, T, input_dim)
        _, (hidden, cell) = self.lstm(x)
        cell = cell.squeeze(0)  # (L, B, hidden_dim)
        
        mean = self.mean_fc(cell)[-1]        # (B, latent_dim)
        log_var = self.log_var_fc(cell)[-1]  # (B, latent_dim)
        return mean, log_var