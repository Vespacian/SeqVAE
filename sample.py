import torch
import matplotlib.pyplot as plt

from model.seqvae import SeqVAE

# Constants, modify for plotting
MODEL_NAME = 'model_gaussian-model_20250102T213123.pth'
MODEL_PATH = f'results/models/{MODEL_NAME}'
NUM_SAMPLES = 4
ELEMENT_DIM = 2
HIDDEN_DIM = 256
LATENT_DIM = 1024
GRAPH_SIZE = 50
DEVICE =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Utility function for sampling from the VAE
def plot_samples(model, num_samples=5):
    # Sample from model
    model.eval()
    sequence_length = GRAPH_SIZE

    with torch.no_grad():
        sampled_sequences = model.sample(num_samples=num_samples, seq_length=sequence_length, device=DEVICE)

    # Plot each sequence
    sampled_sequences_np = sampled_sequences.cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, seq in enumerate(sampled_sequences_np):
        x = seq[:, 0]
        y = seq[:, 1]
        ax.scatter(x, y, marker='o', label=f"Sample {i+1}")

    # Label plot
    ax.set_title(f"Sampled Sequences")
    ax.grid(True)
    ax.legend()

    plt.show()

# Load model and plot
model = SeqVAE(input_dim=ELEMENT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, seq_length=GRAPH_SIZE)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(DEVICE)
print("Model loaded from:", MODEL_PATH)

plot_samples(model, num_samples=NUM_SAMPLES)
