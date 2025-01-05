# Hyperparameter Tuning

This file documents lessons learned and trends observed when hyperparameter tuning the sequential VAE.

## Encoder/Decoder Architecture

- Encoder and decoder should have multiple layers to better capture non-linear dependencies; we currently use 3 layers (2 and 1 perform worse, haven't tried 4)
    - Asymmetric encoder/decoder (ex. different number of layers) does not do as well; a possible explanation is that the encoder still needs the expressivity to capture the nuance
- Encoder output is more effective if we use cell memory as opposed to hidden state at the final timestep
    - Intuition: Cell memory captures long-term dependencies while hidden state captures short-term dependencies, we want a more global view of the level
- Adding a layer to the input of the decoder before decoding is not necessary

## Dimensions

- Hidden dim of 256 with latent dim of 1024 works fairly well
    - Intuition: The latent dim needs to be big enough to capture and allow reconstruction of the information that is output from the encoder and input to the decoder
    - The 1:4 ratio between hidden and latent dim seems fine (1:2 and 4:1 ratios do marginally worse)
    - Smaller combinations (ex. hidden 128 latent 512) do not do as well
- Batch size of 256 works fairly well, smaller values do poorly, haven't experimented much with larger values
- Most experiments are run with epoch size between 2^16 and 2^18

## Regularization and Scheduling

- Dropout value of $p=0.3$ for the encoder and $p=0.7$ for the decoder work fairly well
    - Intuition: Higher value for decoder to allow it to be more robust to going OOD during generation
- Adding slight Gaussian noise to the reconstruction loss performs around the same as not adding noise (or maybe a slight bit better?)
    - We choose to add slight Gaussian noise for sake of correctness; probably gives slight robustness benefits too
- Schedule the KL divergence to weight increasingly heavily in the loss, increasing in about a tanh shape with respect to epoch
    - Current function: `0.5 * (math.tanh((epoch/200) - 2.5) + 1)`

## Distribution Trends

- Uniform distribution is easy to tune and works across a wide range of hyperparameters
- Gaussian mixture distributions are harder to tune, losses go down reasonably but sampling does not always work well
