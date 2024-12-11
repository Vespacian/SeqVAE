import numpy as np
import argparse
import matplotlib.pyplot as plt

from model.seqvae import SeqVAE


def rand_data(batch_size=256, embed_size=2, seq_length=50):
    """generates random integers to try out while training from 0-100

    Args:
        params (batch_size): how many batches should be run
        params (embed_size): specifically, should always be 2 because of (x,y) coordinates
        params (seq_length): how many points to consider per batch

    Returns:
        3D array of integers: shape is (batch_size, embed_size, seq_length) - (n, t, m)
            - in other words, for every batch, there will be a 2D array of x,y points for
            seq_length amount of rows
    """
    
    data = np.random.randint(0, 100, (batch_size, seq_length, embed_size))
    return data

# single loss plot
def plot_loss(loss, num_epochs=50, label="Loss"):
    plt.plot(num_epochs, loss, marker='o', linestyle='-', label=label)
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.title(f'{label} vs Epochs')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # get parameters
    parser = argparse.ArgumentParser(description='Sequential VAE')
    parser.add_argument('--batch-size', type=int, default=256, metavar='BS',
                        help='num batch size (default: 256)')
    parser.add_argument('--seq-length', type=int, default=50, metavar='SL',
                        help='num sequence length (default: 50)')
    parser.add_argument('--num-epochs', type=int, default=10, metavar='EP',
                        help='num epochs (default: 10)')
    args = parser.parse_args()
    
    # Hyperparameters
    batch_size = args.batch_size # n
    seq_length = args.seq_length # t
    num_epochs = args.num_epochs # m
    
    # load data
    data = rand_data(batch_size=batch_size, seq_length=seq_length)
    
    seqvae = SeqVAE(data, batch_size, seq_length)
    
    # start training loop
    for epoch in range(num_epochs):
        pass
    
    plot_loss(0, 0)