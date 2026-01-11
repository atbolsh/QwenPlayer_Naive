# THis file hosts the re-implementation of the transformer, especially for the vision data.
# THis is because the standard torch implementation seems to lack skip connections, which ruins a lot of things
# UPDATE -- torch does have skip connections. The issue was something else. However, I do implement a better version of positional encoding in this file, which helps me a lot

# Based on these sources:
# https://medium.com/thedeephub/building-vision-transformer-from-scratch-using-pytorch-an-image-worth-16x16-words-24db5f159e27
# https://medium.com/thedeephub/building-mae-vision-transformer-from-scratch-using-pytorch-masked-autoencoders-are-scalable-2c2e78e0be02

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

from einops import rearrange


# Best for 1 dimension
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class PositionalEncoding_2D(nn.Module):
    def __init__(self, d_model, num_patches, device = None):
        super(PositionalEncoding_2D, self).__init__()

        if device is None:
            device = 'cpu'
        
        precursor = torch.zeros(num_patches, d_model // 2, device=device)
        position = torch.arange(0, num_patches, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model//2, 2, device=device).float() * -(math.log(10000.0) / (d_model//2)))
        
        precursor[:, 0::2] = torch.sin(position * div_term)
        precursor[:, 1::2] = torch.cos(position * div_term)
        precursor = precursor.unsqueeze(0)

        ########

        pe = torch.zeros(num_patches, num_patches, d_model, device=device)
        pe[:, :, :d_model // 2] += precursor
        pe = pe.transpose(0, 1)
        pe[:, :, d_model // 2:] += precursor

        pe = pe.reshape((num_patches*num_patches), d_model).contiguous()
      
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe


# Stolen from here: https://medium.com/correll-lab/building-a-vision-transformer-model-from-scratch-a3054f707cc6
# Augmented with the Positional Encodings
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()
    
        self.d_model = d_model # Dimensionality of Model
        self.img_size = img_size # Image Size
        self.patch_size = patch_size # Patch Size
        self.n_channels = n_channels # Number of Channels
        self.num_patches = self.img_size // self.patch_size # number of patches on a side
    
        self.linear_project = nn.Conv2d(self.n_channels, self.d_model, kernel_size=self.patch_size, stride=self.patch_size)
           
      # B: Batch Size
      # C: Image Channels
      # H: Image Height
      # W: Image Width
      # P_col: Patch Column
      # P_row: Patch Row
    def forward(self, x):
        x = self.linear_project(x) # (B, C, H, W) -> (B, d_model, P_col, P_row)
   
        x = x.flatten(2) # (B, d_model, P_col, P_row) -> (B, d_model, P)
    
        x = x.transpose(1, 2) # (B, d_model, P) -> (B, P, d_model)
        
        return x


# Begin personal improvization
class PatchEmbeddingTranspose(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()
    
        self.d_model = d_model # Dimensionality of Model
        self.img_size = img_size # Image Size
        self.patch_size = patch_size # Patch Size
        self.n_channels = n_channels # Number of Channels
    
        self.linear_project = nn.ConvTranspose2d(self.d_model, self.n_channels, self.patch_size, stride=self.patch_size, output_padding=0)
    
      # B: Batch Size
      # C: Image Channels
      # H: Image Height
      # W: Image Width
      # P_col: Patch Column
      # P_row: Patch Row
    def forward(self, x):
        x = x.transpose(1, 2) #  (B, P, d_model) -> (B, d_model, P)
     
        B, d, P = x.size()
        P_col = int(math.sqrt(P)) # Assume square image always (change later?)
        P_row = int(P / P_col)
    
        x = x.resize(B, d, P_col, P_row) # (B, d_model, P) -> (B, d_model, P_col, P_row)
    
        x = self.linear_project(x) # (B, d_model, P_col, P_row) -> (B, C, H, W)
       
        return x


class MultiHead(nn.Module):
  def __init__(self, emb_size, num_head):
    super().__init__()
    self.emb_size = emb_size
    self.num_head = num_head
    self.key = nn.Linear(emb_size, emb_size)
    self.value = nn.Linear(emb_size, emb_size)
    self.query = nn.Linear(emb_size, emb_size) 
    self.att_dr = nn.Dropout(0.1)
  def forward(self, x):
    k = rearrange(self.key(x), 'b n (h e) -> b h n e', h=self.num_head)
    q = rearrange(self.query(x), 'b n (h e) -> b h n e', h=self.num_head)
    v = rearrange(self.value(x), 'b n (h e) -> b h n e', h=self.num_head)


    wei = q@k.transpose(3,2)/self.num_head ** 0.5    
    wei = F.softmax(wei, dim=2)
    wei = self.att_dr(wei)

    out = wei@v

    out = rearrange(out, 'b h n e -> b n (h e)')
    return out


class FeedForward(nn.Module):
  def __init__(self, emb_size):
    super().__init__()
    self.ff = nn.Sequential(
        nn.Linear(emb_size, 4*emb_size),
        nn.Linear(4*emb_size, emb_size)
    )
  def forward(self, x):
    return self.ff(x)


class EncBlock(nn.Module):
  def __init__(self,emb_size, num_head):
    super().__init__()
    self.att = MultiHead(emb_size, num_head)
    self.ll =   nn.LayerNorm(emb_size)
    self.dropout = nn.Dropout(0.1)
    self.ff = FeedForward(emb_size)
  def forward(self, x):
    x = x + self.dropout(self.att(self.ll(x)))  # self.att(x): x -> (b , n, emb_size) 
    x = x + self.dropout(self.ff(self.ll(x)))
    return x



