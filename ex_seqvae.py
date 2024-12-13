import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


# -----------------------
# Generate synthetic data
# -----------------------
N = 65536  # number of sequences
T = 50    # sequence length
M = 2     # features (x,y)

class CoordinateDataset(Dataset):
    def __init__(self, data_array):
        self.data = data_array
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)
    



# -----------------------
# Encoder and Decoder
# -----------------------

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

class Decoder(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=32, output_dim=2, seq_length=50):
        super().__init__()
        self.seq_length = seq_length
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z, seq_length=None, drop_prob=0.3):
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

# -----------------------
# SeqVAE that uses the Encoder and Decoder
# -----------------------
class SeqVAE(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, latent_dim=16, seq_length=50):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, seq_length)
        
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

# -----------------------
# Loss function for VAE
# -----------------------
def vae_loss(recon, x, mean, log_var):
    recon_loss = ((recon - x)**2).mean()
    kl_div = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
    return recon_loss + 0 * kl_div, recon_loss, kl_div

# -----------------------
# Training the seqVAE
# -----------------------
hidden_dim = 512
latent_dim = 256
batch_size = 128
epochs = 1

model = SeqVAE(input_dim=M, hidden_dim=hidden_dim, latent_dim=latent_dim, seq_length=T)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()
losses = []
batch_losses = []
for epoch in range(epochs):
    data = np.random.rand(N, T, M).astype(np.float32)
    dataset = CoordinateDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch in dataloader:
        optimizer.zero_grad()
        out, mean, log_var = model(batch)
        loss, rl, kl = vae_loss(out, batch, mean, log_var)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Recon: {rl.item():.4f}, KL: {kl.item():.4f}")
    losses.append(loss.item())

plt.plot(np.arange(epochs), losses)
plt.title(f"epochs by losses: h{hidden_dim}, l{latent_dim}, bs{batch_size}, e{epochs}")
plt.savefig("epochs_losses.png")
plt.close()

plt.plot(batch_losses)
plt.title(f"batch_losses: h{hidden_dim}, l{latent_dim}, bs{batch_size}, e{epochs}")
plt.savefig("batch_losses.png")
plt.close()

# -----------------------
# Sampling from the seqVAE
# -----------------------
model.eval()

print("Plotting ________________________________________________________________________________")
num_samples = 5
sequence_length = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

with torch.no_grad():
    sampled_sequences = model.sample(num_samples=num_samples, seq_length=sequence_length, device=device)
    # sampled_sequences shape: (num_samples, sequence_length, 2)

# Convert to numpy for plotting
sampled_sequences_np = sampled_sequences.cpu().numpy()

# Plot each sequence
fig, ax = plt.subplots(figsize=(8, 6))

for i, seq in enumerate(sampled_sequences_np):
    # seq shape: (T, 2), seq[:,0] is x, seq[:,1] is y
    x = seq[:, 0]
    y = seq[:, 1]

    # Plot the sequence as a line
    ax.scatter(x, y, marker='o', label=f"Sample {i+1}")

# Optionally, add grid, legend, etc.
ax.set_title(f"Sampled Sequences from SeqVAE h{hidden_dim}, l{latent_dim}, bs{batch_size}, e{epochs}")
ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")
ax.grid(True)
ax.legend()

plt.savefig("sampled_seqs.png")
plt.close()
