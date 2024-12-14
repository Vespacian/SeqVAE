import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, latent_dim=16):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.mean_fc = nn.Linear(hidden_dim, latent_dim)
        self.log_var_fc = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        # x: (B, T, input_dim)
        _, (h_n, _) = self.lstm(x)
        h_n = h_n.squeeze(0)  # (B, hidden_dim)
        
        mean = self.mean_fc(h_n)      # (B, latent_dim)
        log_var = self.log_var_fc(h_n) # (B, latent_dim)
        return mean, log_var