import os
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Sequence-to-sequence variational autoencoder, used specifically for generating point configurations")

    # Define and parse arguments
    parser.add_argument('--latent_dim', type=int, default=512, help="Size of the latent dimension of the VAE")
    parser.add_argument('--hidden_dim', type=int, default=1024, help="Size of the hidden dimension of the encoder/decoder")
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs to train on')
    parser.add_argument('--epoch_size', type=int, default=65536, help='Number of instances per epoch during training')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of instances per batch during training')
    parser.add_argument('--graph_size', type=int, default=50, help='Number of points for each graph')

    opts = parser.parse_args(args)

    # Custom arguments
    opts.result_dir = 'results'
    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opts.element_dim = 2 # x, y coordinates for every point

    return opts
