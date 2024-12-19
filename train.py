import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from utils.dataset import CoordinateDataset
from model.seqvae import SeqVAE
from utils.functions import vae_loss
from utils.options import get_options
from utils.distributions import gaussian_mixture_batch


# Utility function for plotting
def plot_losses(losses, opts):
    # Plot data
    epoch_range = np.arange(opts.num_epochs)
    plt.plot(epoch_range, losses['total'], label='Total loss')
    plt.plot(epoch_range, losses['recon'], label='Reconstruction loss')
    plt.plot(epoch_range, losses['kl'], label='KL loss')

    # Label plot
    plt.title(f"Training Loss (h={opts.hidden_dim}, l={opts.latent_dim})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Save plot
    result_dir = os.path.join(opts.result_dir, 'train')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    plt.savefig(os.path.join(result_dir, f'losses_h{opts.hidden_dim}_l{opts.latent_dim}_e{opts.epoch_size}.png'), format='png')
    plt.close()

# Utility function for sampling from the VAE
def plot_samples(model, opts, num_samples=5):
    # Sample from model
    model.eval()
    sequence_length = opts.graph_size

    with torch.no_grad():
        sampled_sequences = model.sample(num_samples=num_samples, seq_length=sequence_length, device=opts.device)

    # Plot each sequence
    sampled_sequences_np = sampled_sequences.cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, seq in enumerate(sampled_sequences_np):
        x = seq[:, 0]
        y = seq[:, 1]
        ax.scatter(x, y, marker='o', label=f"Sample {i+1}")

    # Label plot
    ax.set_title(f"Sampled Sequences (h={opts.hidden_dim}, l={opts.latent_dim})")
    ax.grid(True)
    ax.legend()

    # Save plot
    result_dir = os.path.join(opts.result_dir, 'train')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    plt.savefig(os.path.join(result_dir, f'samples_h{opts.hidden_dim}_l{opts.latent_dim}_e{opts.epoch_size}.png'), format='png')
    plt.close()

# Main run function
def run(opts):
    print("Running with options:")
    print(opts)

    # Define model
    model = SeqVAE(input_dim=opts.element_dim, hidden_dim=opts.hidden_dim, latent_dim=opts.latent_dim, seq_length=opts.graph_size)
    model.to(opts.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Training loop start
    model.train()
    losses = {
        'total': [],
        'recon': [],
        'kl': []
    }
    
    for epoch in range(opts.num_epochs):
        # Train with uniform random data
        # data = np.random.rand(opts.epoch_size, opts.graph_size, opts.element_dim).astype(np.float32)
        
        # from gaussian
        data = np.zeros((opts.epoch_size, opts.graph_size, opts.element_dim), dtype=np.float32)
        # for i in range(opts.epoch_size):
        data = gaussian_mixture_batch(opts.batch_size, opts.graph_size, cdist=50)
        # print(data[epoch])
        
        # sorting data by increasing values of x
        sorted_indicies = np.argsort(data[:, :, 0], axis=1)
        sorted_data = np.take_along_axis(data, sorted_indicies[:, :, None], axis=1)
        # print(sorted_data)
        
        # confirm data was sorted properly
        x_vals = sorted_data[:, :, 0] 
        assert np.all(np.diff(x_vals, axis=1) >= 0), "x values are not sorted in non decreasing order"
        
        dataset = CoordinateDataset(sorted_data)
        dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, pin_memory=True)
        for batch in dataloader:
            batch = batch.to(opts.device)
            
            optimizer.zero_grad()
            out, mean, log_var = model(batch)
            loss, rl, kl = vae_loss(out, batch, mean, log_var, epoch)
            loss.backward()
            optimizer.step()
        
        # Save and log losses
        losses['total'].append(loss.item())
        losses['recon'].append(rl.item())
        losses['kl'].append(kl.item())
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Recon: {rl.item():.4f}, KL: {kl.item():.4f}")

    # Plotting results
    plot_losses(losses, opts)
    plot_samples(model, opts)
    print("Training complete and plots saved")

# Program entrypoint
if __name__ == "__main__":
    run(get_options())
