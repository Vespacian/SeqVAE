import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from utils.coord_dataset import CoordinateDataset
from model.seqvae import SeqVAE
from utils.functions import vae_loss


def plot_losses(hidden_dim, latent_dim, epochs, losses, batch_losses):
    plt.plot(np.arange(epochs), losses)
    plt.title(f"epochs losses: h-{hidden_dim}, l-{latent_dim}, bs-{batch_size}, e-{epochs}")
    plt.xlabel("epochs")
    plt.ylabel("losses")
    plt.savefig("epochs_losses.png")
    plt.close()

    plt.plot(batch_losses)
    plt.title(f"batch losses: h-{hidden_dim}, l-{latent_dim}, bs-{batch_size}, e-{epochs}")
    plt.xlabel("batches")
    plt.ylabel("losses")
    plt.savefig("batch_losses.png")
    plt.close()

# Sampling from the seqVAE
def sampling():
    model.eval()

    print("Plotting ________________________________________________________________________________")
    num_samples = 5
    sequence_length = 50

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
    ax.set_title(f"Sampled Sequences from SeqVAE h-{hidden_dim}, l-{latent_dim}, bs-{batch_size}, e-{epochs}")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True)
    ax.legend()

    plt.savefig("sampled_seqs.png")
    plt.close()

# expects np array
# expected elements: epochs, loss val, recon loss, kl div
def plot_progress(progress):
    plt.plot(progress[:, 0], progress[:, 1], label='Total loss', marker='o')
    plt.plot(progress[:, 0], progress[:, 2], label='Recon loss', marker='o')
    plt.plot(progress[:, 0], progress[:, 3], label='KL Div', marker='o')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Progress')
    plt.legend()
    plt.grid(True)
    
    plt.savefig("progress.png")
    plt.close()


# _______________________________________________________________________________________________________
# Dims for generating random data
N = 65536  # number of sequences
T = 50    # sequence length
M = 2     # features (x,y)

# Hypterparameters
hidden_dim = 512
latent_dim = 1024
batch_size = 128
epochs = 2

# init model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SeqVAE(input_dim=M, hidden_dim=hidden_dim, latent_dim=latent_dim, seq_length=T)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop start
model.train()
losses = []
batch_losses = []
progress = []
for epoch in range(epochs):
    # testing with random data
    data = np.random.rand(N, T, M).astype(np.float32)
    dataset = CoordinateDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    for batch in dataloader:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        out, mean, log_var = model(batch)
        loss, rl, kl = vae_loss(out, batch, mean, log_var, epoch)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    
    progress.append((epoch+1, loss.item(), rl.item(), kl.item()))
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Recon: {rl.item():.4f}, KL: {kl.item():.4f}")
    losses.append(loss.item())

# Plotting results
plot_progress(np.array(progress))
plot_losses(hidden_dim, latent_dim, epochs, losses, batch_losses)
sampling()

