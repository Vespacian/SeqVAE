import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from .embedding import Embedding
from .encoder import Encoder
from .decoder import Decoder

from utils.functions import kld_coef

class SeqVAE(nn.Module):
    def __init__(self, params):
        super(SeqVAE, self).__init__()
        
        # (t, m)
        self.params = params
        
        self.encoder = Encoder(self.params)
        self.embedding = Embedding(self.params)
        self.decoder = Decoder(self.params)
        
        # formulas to use in forward pass
        self.context_to_mu = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size) # TODO: what is latent_variable_size?
        self.context_to_logvar = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)
        
    
    def forward(self, drop_prob, coordinates=None):
        """
        :param initial_state: initial state of decoder rnn in order to perform sampling

        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout

        :param z: context if sampling is performing

        :return: unnormalized logits of sentence words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
                 
        return: distribution probabilities with shape (n, t, m)
        """

        # embed the input coordinates
        if coordinates is None:
            print("need to enter coordinates")
            return 0, 0, 0
        else:
            batch_size, embedding_size, seq_len = coordinates.size()

            # encoder_input = self.embedding(encoder_word_input, encoder_character_input)
            # flatten input for embedding (n, t, m) -> (n, t x m)
            encoder_input = self.embedding(coordinates.view(batch_size, -1))
            context = self.encoder(encoder_input)

            mu = self.context_to_mu(context)
            logvar = self.context_to_logvar(context)
            std = t.exp(0.5 * logvar)
            
            # check if possible on gpu
            z = t.randn([batch_size, self.params.latent_variable_size])
            device = t.device("cuda" if t.cuda.is_available() else "cpu")
            z = z.to(device)
            
            # reparameterization trick
            z = z * std + mu

            # KL divergence loss
            kld = (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean().squeeze()

        # decode the latent representation
        # shape now becomes (batch_size, seq_len, latent_variable_size)
        decoder_input = z.unsqueeze(1).repeat(1, seq_len, 1)
        out, final_state = self.decoder(decoder_input, z, drop_prob)

        return out, final_state, kld

    def learnable_parameters(self):
        # word_embedding is constant parameter thus it must be dropped from list of parameters for optimizer
        return [p for p in self.parameters() if p.requires_grad]

    def trainer(self, optimizer, batch_loader):
        def train(i, batch_size, use_cuda, dropout):
            input = batch_loader.next_batch(batch_size, 'train')
            input = [t.from_numpy(np.array(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target] = input

            logits, _, kld = self(dropout,
                                  encoder_word_input, encoder_character_input,
                                  decoder_word_input, decoder_character_input,
                                  z=None)

            logits = logits.view(-1, self.params.word_vocab_size)
            target = target.view(-1)
            cross_entropy = F.cross_entropy(logits, target)

            loss = 79 * cross_entropy + kld_coef(i) * kld

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return cross_entropy, kld, kld_coef(i)

        return train

    def validater(self, batch_loader):
        def validate(batch_size, use_cuda):
            input = batch_loader.next_batch(batch_size, 'valid')
            input = [t.from_numpy(var) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target] = input

            # change constructor
            logits, _, kld = self(0.,
                                  encoder_word_input, encoder_character_input,
                                  decoder_word_input, decoder_character_input,
                                  z=None)

            logits = logits.view(-1, self.params.word_vocab_size)
            target = target.view(-1)

            cross_entropy = F.cross_entropy(logits, target)

            return cross_entropy, kld

        return validate

    def sample(self, batch_loader, seq_len, seed, use_cuda):
        seed = t.from_numpy(seed).float()
        if use_cuda:
            seed = seed.cuda()

        decoder_word_input_np, decoder_character_input_np = batch_loader.go_input(1)

        decoder_word_input = t.from_numpy(decoder_word_input_np).long()
        decoder_character_input = t.from_numpy(decoder_character_input_np).long()


        if use_cuda:
            decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()

        result = ''

        initial_state = None

        for i in range(seq_len):
            logits, initial_state, _ = self(0., None, None,
                                            decoder_word_input, decoder_character_input,
                                            seed, initial_state)

            logits = logits.view(-1, self.params.word_vocab_size)
            prediction = F.softmax(logits)

            word = batch_loader.sample_word_from_distribution(prediction.data.cpu().numpy()[-1])

            if word == batch_loader.end_token:
                break

            result += ' ' + word

            decoder_word_input_np = np.array([[batch_loader.word_to_idx[word]]])
            decoder_character_input_np = np.array([[batch_loader.encode_characters(word)]])

            decoder_word_input = t.from_numpy(decoder_word_input_np).long()
            decoder_character_input = t.from_numpy(decoder_character_input_np).long()
            
            if use_cuda:
                decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()

        return result

    