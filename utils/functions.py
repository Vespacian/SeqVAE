import torch

# -----------------------
# Loss function for VAE
# -----------------------
def vae_loss(recon, x, mean, log_var, epoch):
    recon_loss = ((recon - x)**2).mean()
    kl_div = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
    # if epoch < 50:
    #     return recon_loss + 0 * kl_div, recon_loss, kl_div
    return recon_loss + 0.01 * kl_div, recon_loss, kl_div