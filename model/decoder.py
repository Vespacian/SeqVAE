import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=32, output_dim=2, seq_length=50):
        super().__init__()
        self.seq_length = seq_length
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z, seq_length=None, drop_prob=0): # drop prob used to be 0.3
        if seq_length is None:
            seq_length = self.seq_length
        
        z = F.dropout(z, drop_prob)
        
        # z: (B, latent_dim)
        # Repeat z for each time step: (B, T, latent_dim)
        z_expanded = z.unsqueeze(1).repeat(1, seq_length, 1)
        
        h, _ = self.lstm(z_expanded)
        # h: (B, T, hidden_dim)
        out = self.fc_out(h)  # (B, T, output_dim)
        return out