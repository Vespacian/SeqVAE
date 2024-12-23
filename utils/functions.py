import torch
import math

def kl_scheduler(epoch):
    return 0.5 * (math.tanh((epoch/100) - 2.5) + 1)

# Loss function for VAE
def vae_loss(recon, x, mean, log_var, epoch):
    recon_loss = ((recon - x)**2).mean()
    kl_div = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
    return recon_loss + kl_scheduler(epoch) * kl_div, recon_loss, kl_div
