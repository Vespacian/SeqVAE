# SeqVAE

## Setup

Create a Python environment and install the required dependencies:

```
pip install -r requirements.txt
```

## Training

To start training, run `train.py`. Run `python train.py --help` for a full list of customizable options.

Example command:

```
python train.py --epoch_size 65536 --batch_size 256 --latent_dim 256 --hidden_dim 1024 --num_epochs 150
```
