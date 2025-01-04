import torch
import matplotlib.pyplot as plt
import math

from model.seqvae import SeqVAE

# Constants, modify for plotting
MODEL_NAME = 'model_gaussian-model-0.7dp_20250103T120759.pth'
MODEL_PATH = f'results/models/{MODEL_NAME}'
NUM_SAMPLES = 12
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

    # Utility function for plotting subplot
    def subplot_embedding(subplot, graph, title):
        subplot.scatter(graph[:,0], graph[:,1], color='blue')
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.set_title(title)

    # Plot each sequence
    sampled_sequences_np = sampled_sequences.cpu().numpy()
    WIDTH = 4
    HEIGHT = math.ceil(num_samples / WIDTH)

    fig, axs = plt.subplots(HEIGHT, WIDTH, figsize=(WIDTH * 3, HEIGHT * 3))
    plt.subplots_adjust(hspace=0.1)
    plt.tight_layout(rect=[0.01, 0, 0.97, 0.98])

    for i, sample in enumerate(sampled_sequences_np):
        idx1 = i // WIDTH
        idx2 = i % WIDTH
        subplot_embedding(axs[idx1, idx2], sample, f"Sample {i+1}")

    plt.show()

# Load model and plot
model = SeqVAE(input_dim=ELEMENT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, seq_length=GRAPH_SIZE)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.to(DEVICE)
print("Model loaded from:", MODEL_PATH)

plot_samples(model, num_samples=NUM_SAMPLES)
