import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.distributions import Categorical

import math
import copy

from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset

from .custom_transformer import *

## Memory processing

# class Memory Processor
# consumes the memory tensor, uses positional encoding on it, and uses the Decoder architecture (maybe 1 encoder layer?) to produce a vector based on the text and image encodigns
# THis vector is then used as context for the others (eg for answering questions or recalling saved images).
class MemoryProcessor(nn.Module):
    def __init__(self, sequence_length=128, embed_dim=768, num_heads=6, num_layers=8, dropout=0.1, norm_first=False):
        super(MemoryProcessor, self).__init__()
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.sqrt_embed_dim = math.sqrt(embed_dim)
        self.embed = nn.Sequential(
            PositionalEncoding(embed_dim, sequence_length),
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=0.1),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True, norm_first=norm_first,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.text_img_marker = PositionalEncoding(embed_dim, 2) # extend to num_canvases? Or some other processing like that?

        # Really necessary? Think
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.LayerNorm(embed_dim * 4),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(embed_dim * 4, embed_dim),
        )

        # Convenient tensor:
        self.consecutive_indeces = torch.LongTensor(list(range(self.sequence_length))).to(self.get_device())

    def get_device(self):
        return self._modules['fc'][0].weight.device # this function should be part of nn.Module, honestly
   
    # Accepts already encoded inputs (dimension 768)
    def forward(self, x, text_context=None, img_context=None):
        x = self.embed(x)
        if (text_context is None) and (img_context is None):
            context = x
        elif (not (text_context is None)) and (img_context is None):
            context = text_context
        elif (text_context is None) and (not (img_context is None)):
            context = img_context
        else:
            context = torch.cat((text_context + self.text_img_marker.pe[:, :1], img_context + self.text_img_marker.pe[:, 1:]), dim=1) # default mode
        x = self.decoder(x, context)
        return self.fc(x)


# class Memory Encoder
# consumes arbitrary input (standardized to one text and 3 canvases) + memory; probably uses markers for 1st tensor, 2nd tensor, 3rd tensor, etc.
# Produces 4 vectors of length 768 (or 1?? consier)
# These are fed through the 'forget' gates and added to existing memory
# This class will have the infamous 'saved forget gate'
class MemoryEncoder(nn.Module):
    # I made it have fewer heads and layers for compactness; check later if this is enough
    def __init__(self, new_tokens=1, embed_dim=768, num_heads=3, num_layers=4, dropout=0.1, norm_first=False):
        super(MemoryEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.new_tokens = new_tokens
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True, norm_first=norm_first,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        
        self.fc_prep = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.LayerNorm(embed_dim * 4),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout)
        ) 
        self.fcs = nn.Sequential(*[nn.Linear(4*embed_dim, embed_dim) for i in range(self.new_tokens)]) # different final layer for each token used

    def get_device(self):
        return self._modules['decoder']._modules['layers'][0].linear1.weight.device # this function should be part of nn.Module, honestly
    
    # x is the text; context is the images
    def forward(self, x, context=None):
        if context is None:
            context = x
        x = self.decoder(x, context)
        prep = self.fc_prep(x) # 4 times larger internal dim
        results = torch.zeros((x.size()[0], self.new_tokens, x.size()[2]), dtype=x.dtype, device=x.device)
        for i in range(self.new_tokens):
            results[:, i, :] = torch.sum(self.fcs[i](prep), dim=1)
        return results


# class Memory
# Tensor storage
# initialization, addition of the new 1-token thing
# new_tokens MUST factor neatly into mem_size
class Memory(nn.Module):
    def __init__(self, mem_size=128, new_tokens=1, vector_dim=768):
        super(Memory, self).__init__()

        self.mem_size = mem_size
        self.vector_dim = vector_dim
        self.new_tokens = new_tokens
        self.repetitions = int(mem_size // new_tokens)

        starter = 1.0 / torch.arange(1, self.repetitions + 1, 1)
        scales = starter.unsqueeze(0).repeat(new_tokens, 1).T.flatten().contiguous()
        scales = scales.unsqueeze(0).unsqueeze(2).contiguous()

        memory = torch.zeros(1, mem_size, vector_dim)
        self.empty = True

        self.register_buffer('memory', memory)
        self.register_buffer('scales', scales)

    def is_empty(self):
        return self.empty

    def get_device(self):
        return self.scales.device

    def soft_reset(self):
        self.memory = self.memory.detach()
        self.scales = self.scales.detach()

    # tokens has shape (batches, new_tokens, 768). Even if batches is 1
    def remember(self, tokens):
        # last layer forgot to unsqueeze
        if len(tokens.size()) == 2:
            tokens = tokens.unsqueeze(1).repeat(1, self.new_tokens, 1)
        tokens = tokens.repeat(1, self.repetitions, 1)
        if self.is_empty():
            self.memory = tokens + self.memory # have to start somewhere; the addition makes the device stay in place
        else:
            self.memory = self.memory * (1.0 - self.scales) + (self.scales * tokens)
        self.empty = False

    def forward(self, tokens):
        self.remember(tokens)
        return self.memory


