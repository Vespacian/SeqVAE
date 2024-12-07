import numpy as np
import torch as t
import torch.nn as nn

from .embedding import Embedding
from .encoder import Encoder
from .decoder import Decoder

class SeqVAE(nn.Module):
    def __init__(self, params):
        super(SeqVAE, self).__init__()
        
        self.params = params
        
        self.embedding = Embedding()
        self.encoder = Encoder()
        # functions? context to mu, context to logvar?
        self.decoder = Decoder()
        
    
    def forward():
        pass
    