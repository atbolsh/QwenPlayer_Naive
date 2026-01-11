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

# Main workhorse, image to embedding
# Default are the ViT parameters: https://arxiv.org/pdf/2010.11929 page 5
class ImageTransformerEncoder(nn.Module):
    def __init__(self, num_channels=3, num_patches=16, img_size=224, embed_dim=768, num_heads=12, num_layers=6, output_dim=768, dropout=0.1, norm_first=False):
        super().__init__()
        self.patch_num = num_patches
        self.sequence_length = num_patches * num_patches
        self.img_size = img_size
        self.patch_size = img_size // num_patches
        self.embed_dim = embed_dim
        self.sqrt_embed_dim = math.sqrt(embed_dim)
        self.embed = nn.Sequential(
            PatchEmbedding(embed_dim, self.img_size, self.patch_size, num_channels),
            PositionalEncoding_2D(embed_dim, num_patches),
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=0.1),
        )

        self.pe = self.embed[1]
        # I may be doing this wrong; I may need 1024 * 3 instead of 768 here, but I think this will do for now.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True, norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.post_norm = nn.LayerNorm(embed_dim)
        # Convenient tensor:
        self.consecutive_indeces = torch.LongTensor(list(range(self.sequence_length))).to(self.get_device())

    def get_device(self):
        return self.pe.pe.device
    
    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x) # x + self.encoder(x)
#        return self.post_norm(self.pe(x)) # add position information at the end, too
        return self.post_norm(x)

# At the end, we have an emedding for every patch.

class ImageTransformerDecoder(nn.Module):
    def __init__(self, num_channels=3, num_patches=16, img_size=224, embed_dim=768, num_heads=12, num_layers=3, output_dim=768, dropout=0.01, norm_first=False):
        super().__init__()
        self.patch_num = num_patches
        self.sequence_length = num_patches * num_patches
        self.img_size = img_size
        self.patch_size = img_size // num_patches
        self.embed_dim = embed_dim
        self.sqrt_embed_dim = math.sqrt(embed_dim)
        # I may be doing this wrong; I may need 1024 * 3 instead of 768 here, but I think this will do for now.
        self.pe = PositionalEncoding_2D(embed_dim, num_patches)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True, norm_first=norm_first,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.linear_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.LayerNorm(embed_dim),
            PatchEmbeddingTranspose(embed_dim, self.img_size, self.patch_size, num_channels)
        )
        # Convenient tensor:
        self.consecutive_indeces = torch.LongTensor(list(range(self.sequence_length))).to(self.get_device())

    def get_device(self):
        return self.pe.pe.device
    
    def forward(self, x, context = None):
        if context is None:
            context = x
        x = self.decoder(x, context)
        return self.linear_layer(x)

#### Lang model

def generate_src_mask(sz, device):
    return torch.triu(torch.full((sz, sz), True, device=device), diagonal=1) # Changing backto =1 to nan's

def generate_src_padding_mask(x):
    mask = (x == 0) #torch.logical_and((x == 0), (x != 4)) # I'm going to use the <mask> tag as the placeholder for the images themselves; <img><mask> is the standard now.
    mask[:, 0] = False # include first element
    return mask

class PositionalEmbedding(nn.Module):
    def __init__(self, sequence_length, embed_dim):
        super().__init__()
        self.sqrt_embed_dim = math.sqrt(embed_dim)
        self.pos_embed = nn.Parameter(torch.empty((1, sequence_length, embed_dim)))
        nn.init.uniform_(self.pos_embed, -1.0, 1.0)
    
    def forward(self, x):
        return x * self.sqrt_embed_dim + self.pos_embed[:,:x.size(1)]


class SentenceTransformerEncoder(nn.Module):
    def __init__(self, sequence_length=32, num_embed=10000, embed_dim=768, pad_idx=0, num_heads=6, num_layers=8, dropout=0.1, norm_first=False):
        super().__init__()
        self.vocab_size = num_embed
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.sqrt_embed_dim = math.sqrt(embed_dim)
        self.embed = nn.Sequential(
            nn.Embedding(num_embed, embed_dim, pad_idx),
            PositionalEncoding(embed_dim, sequence_length),
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=0.1),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True, norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Convenient tensor:
        self.consecutive_indeces = torch.LongTensor(list(range(self.sequence_length))).to(self.get_device())

    def get_device(self):
        return self._modules['encoder']._modules['layers'][0].linear1.weight.device # this function should be part of nn.Module, honestly
    
    def forward(self, x, src_attention_mask=None, src_key_padding_mask=None):
        x = self.embed(x)
        x = self.encoder(x, mask=src_attention_mask, src_key_padding_mask=src_key_padding_mask)
        return x

# THis is useful as a value function
# Consumes only intermediate representations (768 dim), dumps 1 value per batch
class IntermediateTransformerScorer(nn.Module):
    # I made it have fewer heads and layers for compactness; check later if this is enough
    def __init__(self, sequence_length=32, embed_dim=768, pad_idx=0, num_heads=3, num_layers=4, dropout=0.1, norm_first=False):
        super().__init__()
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True, norm_first=norm_first,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # Convenient tensor:
        self.consecutive_indeces = torch.LongTensor(list(range(self.sequence_length))).to(self.get_device())
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.LayerNorm(embed_dim * 4),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(embed_dim * 4, 1),
        )

    def get_device(self):
        return self._modules['decoder']._modules['layers'][0].linear1.weight.device # this function should be part of nn.Module, honestly
    
    def forward(self, x, context=None):
        if context is None:
            context = x
        x = self.decoder(x, context)
        x = self.fc(x)
        x = torch.sum(x, dim=1) # sum accross the sequence dimension. Replace with average?
        return x

class SentenceTransformerDecoder(nn.Module):
    def __init__(self, sequence_length=32, num_embed=10000, embed_dim=768, pad_idx=0, num_heads=6, num_layers=4, dropout=0.1, norm_first=False):
        super().__init__()
        self.vocab_size = num_embed
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.sqrt_embed_dim = math.sqrt(embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True, norm_first=norm_first,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.LayerNorm(embed_dim * 4),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(embed_dim * 4, num_embed),
        )
        # Convenient tensor:
        self.consecutive_indeces = torch.LongTensor(list(range(self.sequence_length))).to(self.get_device())

    def get_device(self):
        return self._modules['decoder']._modules['layers'][0].linear1.weight.device # this function should be part of nn.Module, honestly
    
    def forward(self, x, context=None, return_full=False, tgt_mask=None, tgt_key_padding_mask=None, memory_mask=None, memory_key_padding_mask=None):
        if context is None:
            context = x
            memory_mask = tgt_mask
            memory_key_padding_mask = memory_key_padding_mask
        x = self.decoder(x, context, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_mask=memory_mask, memory_key_padding_mask=memory_key_padding_mask)
        if return_full:
            x = self.fc(x)
            x = x.permute(0, 2, 1)
            return x
        else: # only last character; perhaps useful for per-character generation. May be replaced with something else, like torch.sum(dim=1)
            x = x[:, -1, :]
            x = self.fc(x)
            return x # one logit vector for every batch member.


class SolitaryValueFunc(nn.Module):
    def __init__(self, vocab_size=7):
        super().__init__()
        self.text_enc = SentenceTransformerEncoder(num_embed=vocab_size)
        self.dopamine = IntermediateTransformerScorer() # for RL; not yet tested, use later

    # I will keep this as the default dopamine signal; may spin some other 'raw default' types later.
    def evaluate_text(self, text_batch, context=None, text_gradient = True):
        if text_gradient:
            text_encoding = self.text_enc(text_batch)
        else:
            with torch.no_grad():
                text_encoding = self.text_enc(text_batch)
        return self.dopamine(text_encoding, context)

    def forward(self, text, context=None):
        return self.evaluate_text(text, context)

# Default class. Will get built out into full agent brain
class DefaultAgentBrain(nn.Module):
    def __init__(self, vocab_size=10000):
        super().__init__()
        self.img_enc = ImageTransformerEncoder()
        self.img_dec = ImageTransformerDecoder()
        self.text_enc = SentenceTransformerEncoder(num_embed=vocab_size)
        self.text_dec = SentenceTransformerDecoder(num_embed=vocab_size)
        self.dopamine = IntermediateTransformerScorer() # for RL; not yet tested, use later
        # sometimes it makes sense to compare images; this embedding helps distinguish them
        # Two images; 768 values as the hidden layer dim
        # I may use more than 2 images later.
        self.img_tagging = nn.Parameter(torch.empty((1, 2, 768)))
        nn.init.uniform_(self.img_tagging, -1.0/math.sqrt(768), 1.0/math.sqrt(768)) 

    def get_device(self):
        return self.img_enc.get_device()

    def get_masks(self, text_batch, use_masks=True):
        if use_masks:
            src_attention_mask = generate_src_mask(text_batch.size(1), text_batch.device)
            src_key_padding_mask = generate_src_padding_mask(text_batch)
        else:
            src_attention_mask = None
            src_key_padding_mask = None
        return src_attention_mask, src_key_padding_mask

    def get_text_encoding(self, text_batch, src_attention_mask, src_key_padding_mask):
        return self.text_enc(text_batch, src_attention_mask=src_attention_mask, src_key_padding_mask=src_key_padding_mask)

    def get_text_decoding(self, text_encoding, src_attention_mask, src_key_padding_mask, context=None, return_full=True):
        return self.text_dec(text_encoding, context, return_full=return_full, tgt_mask=src_attention_mask, tgt_key_padding_mask=src_key_padding_mask)

    def forward(self, text_batch, img_batch=None, ret_imgs=False, return_full=True, use_masks=True):
        src_attention_mask, src_key_padding_mask = self.get_masks(text_batch, use_masks)
        text_encoding = self.get_text_encoding(text_batch, src_attention_mask, src_key_padding_mask)
        if img_batch is None:
            img_context = text_encoding # just feed the text features back to itself.
        else:
            img_context = self.img_enc(img_batch)
        text_probs = self.get_text_decoding(text_encoding, src_attention_mask, src_key_padding_mask, img_context, return_full)
        if not ret_imgs:
            return text_probs
        else:
            if img_batch is None:
                batches = text_batch.size()[0]
                img_encoding = torch.zeros((batches, 256, 768), device=self.get_device())
            else:
                img_encoding = img_context
            img_reconstruction = self.img_dec(img_encoding, text_encoding)
            return text_probs, img_reconstruction

    # used for training question-answering; text_batch_in has SAME shape, but answers zero-ed out.
    # slower than straight-up text training. ONly use if you are ALSO training image reconstruction.
    # otherwise, just use the old 'forward' and only compute the gradient with respect to the text transformers
    def qa_forward(self, text_batch_in, text_batch_out, img_batch=None, ret_imgs=False, return_full=True, use_masks=True):
        src_attention_mask_in, src_key_padding_mask_in = self.get_masks(text_batch_in, use_masks)
        src_attention_mask_out, src_key_padding_mask_out = self.get_masks(text_batch_out, use_masks)
        text_encoding_in = self.get_text_encoding(text_batch_in, src_attention_mask_in, src_key_padding_mask_in)
        text_encoding_out = self.get_text_encoding(text_batch_out, src_attention_mask_out, src_key_padding_mask_out)
        if img_batch is None:
            img_context = text_encoding_in # just feed the text features back to itself.
        else:
            img_context = self.img_enc(img_batch)
        # when generating the answer string, each token still sees all the preceding tokens, and NO others
        # this is valid; the answer token (eg 'left') will still have to be inferred
        # I will reassess whether this is valid for longer answers
        text_probs = self.get_text_decoding(text_encoding_out, src_attention_mask_out, src_key_padding_mask_out, img_context, return_full)
        if not ret_imgs:
            return text_probs
        else:
            if img_batch is None:
                batches = text_batch.size()[0]
                img_encoding = torch.zeros((batches, 256, 768), device=self.get_device())
            else:
                img_encoding = img_context
            img_reconstruction = self.img_dec(img_encoding, text_encoding_in) # CRITICAL! Only the 'question', not the 'answer', is used when computing the image!
            return text_probs, img_reconstruction

    def img_autoencoder(self, img_batch, context = None):
        img_encoding = self.img_enc(img_batch)
        if context is None:
            context = img_encoding
        return self.img_dec(img_encoding, context)

    def sentence_autoencoder(self, text_batch, context = None, return_full=True, use_masks=False):
        src_attention_mask, src_key_padding_mask = self.get_masks(text_batch, use_masks)
        text_encoding = self.get_text_encoding(text_batch, src_attention_mask, src_key_padding_mask)
#        print(text_encoding)
        # These lines not needed; this chance is covered more smoothly by the conditions within text_dec
#        if context is None:
#            context = text_encoding
        return self.get_text_decoding(text_encoding, src_attention_mask, src_key_padding_mask, context, return_full)

    # I will keep this as the default dopamine signal; may spin some other 'raw default' types later.
    # can sometimes train the dopamine layer alone, without any text or image gradients.
    # Must then set the img_gradient and text_gradient settings to False, or you get a memory leak
    def evaluate_text(self, text_batch, img_batch=None, img_gradient=True, text_gradient=True):
        if text_gradient:
            text_encoding = self.text_enc(text_batch)
        else:
            with torch.no_grad():
                text_encoding = self.text_enc(text_batch)
        if img_batch is None:
            return self.dopamine(text_encoding)
        else:
            if img_gradient:
                context = self.img_enc(img_batch)
            else:
                with torch.no_grad():
                    context = self.img_enc(img_batch)
            return self.dopamine(text_encoding, context)

    # useful for comparing images
    # I may later switch to using this by default
    def two_img_context(self, img1_batch, img2_batch):
        batches = img1_batch.size()[0]
        context = torch.zeros((batches, 256*2), device = self.get_device())
        img1_encoding = self.img_enc(img1_batch)
        img2_encoding = self.img_enc(img2_batch)
        context[:, :256] += (img1_encoding + self.img_tagging[0])
        context[:, 256:] += (img2_encoding + self.img_tagging[1])
        return context

    def select(self, logits, temp=0.0, ret_all = True, temp_eps = 1e-4):
        # logits is a vector batch x vocab size
        # ret_all means return log probs and entropy; that's the assumed behavior.
        # This is a different input than model_declarations_v2: I am assuming batch generation, and the logits come from different batches
        if temp < temp_eps:
            preds = torch.argmax(logits, dim=1)
            if not ret_all:
                return preds
            log_probs = torch.max(F.log_softmax(logits, dim=1), dim=1)
            dist = Categorical(logits = logits)
            entropy = dist.entropy() # should be length batch
            return preds, log_probs, entropy
        else: # very convenient. Maybe add a torch.no_grad()? Dunno
            dist = Categorical(logits = logits / temp)
            preds = dist.sample()
            if not ret_all:
                return preds
            entropy = dist.entropy()
            log_probs = dist.log_prob(preds)
            return preds, log_probs, entropy

    def extend(self, seed, is_terminated, context=None, temp=1.0, ret_all=True, temp_eps = 1e-4):
        s = seed.size()
        output = torch.zeros((s[0], s[1] +1), dtype = torch.long, device = seed.device)
        output[:, :-1] += seed
        # Process and return only the logits for the final character. Use the provided (visual) context
        # the final zero is ignored in the computation due to the mask
        logits = self.sentence_autoencoder(output, context, use_masks=True, return_full=False)
        if not ret_all:
            preds = self.select(logits, temp, ret_all, temp_eps)
        else:
            preds, log_probs, entropy = self.select(logits, temp, ret_all, temp_eps)
        preds = preds * torch.logical_not(is_terminated) # set all past-terminated actions / tokens to 0
        output[:, -1] += preds
        #print(preds==2)
        is_terminated = torch.logical_or(is_terminated, (preds==2))
        if not ret_all:
            return output, preds, is_terminated
        else:
            return output, preds, log_probs, entropy, is_terminated # maybe also multiply by is_terminated? Dunno.

    def generate(self, x=None, context=None, maxlen = None, temp=1.0, ret_all=True, temp_eps = 1e-4, default_batches = 1):
        if maxlen is None:
            maxlen = self.text_enc.sequence_length
        if x is None:
            x = torch.zeros((default_batches, 1), device=self.get_device(), dtype=torch.long)
        if ret_all:
            lp = torch.zeros((default_batches, 1), device=self.get_device()) # default dtype
            ent = torch.zeros((default_batches, 1), device=self.get_device())
        batches, _ = x.size()
        is_terminated = torch.zeros(batches, dtype=torch.bool, device=self.get_device()) # none are terminated initially
        if ret_all:
            lp = torch.zeros((batches, 1), device=self.get_device()) # default dtype
            ent = torch.zeros((batches, 1), device=self.get_device())
        firstGone = False
        while (x.size()[1] < maxlen) and (not torch.all(is_terminated)):
            if ret_all:
                x, _, newlp, newent, is_terminated = self.extend(x, is_terminated, context, temp, ret_all, temp_eps)
                if firstGone: # so, in all cases except the first value
                    lp = F.pad(lp, (0, 1))
                    ent = F.pad(ent, (0, 1))
                else:
                    firstGone=True
                lp[:, -1] += newlp
                ent[:, -1] += newent
            else:
                x, _, is_terminated = self.extend(x, is_terminated, context, temp, ret_all, temp_eps)
        if ret_all:
            return x, lp, ent
        else:
            return x

#    def compute_probabilities(self, x, seed_offset=0, context=None, temp=1.0):
#        """Given sentences x, possibly computed by another model, compute the logpas and entropies for all the values chosen.
#           (Notice that we are talking about the values ALREADY chosen, not the choices this model would make.)"""
#        batches, total_len = x.size()
#        gen_len = total_len - seed_offset
#        logpas = torch.zeros(batches, gen_len, device=x.device)
#        entropies = torch.zeros(batches, gen_len, device=x.device)
#        for i in range(gen_len):
#            cutoff = i + seed_offset
#            logits = self.sentence_autoencoder(x[:, :cutoff], context, use_masks=True, return_full=False)
#            dist = Categorical(logits = logits / temp)
#            entropies[:, i] = dist.entropy()
#            logpas[:, i] = dist.log_prob(x[:, cutoff])
#        return logpas, entropies

    # SINGLE only returns probs for final val; multi returns all probs (assumes identical contexts)
    def compute_probabilities(self, x, seed_offset=1, context=None, temp=1.0, single=False):
        if single:
            return self._compute_probabilities_SINGLE(x, context, temp)
        else:
            return self._compute_probabilities_MULTI(x, seed_offset, context, temp)

    def _compute_probabilities_MULTI(self, x, seed_offset, context=None, temp=1.0):
        """Given sentences x, possibly computed by another model, compute the logpas and entropies for all the values chosen.
           (Notice that we are talking about the values ALREADY chosen, not the choices this model would make.)"""
        batches, total_len = x.size()
        gen_len = total_len - seed_offset
        logits = self.sentence_autoencoder(x, context=context, use_masks=True, return_full=True)[:, :, seed_offset-1:-1] # should be batches x tokens x genlen
        logits = logits.transpose(1, 2) # bathes x genlen x tokens
        logits = logits.reshape((batches*gen_len, self.text_dec.vocab_size)) # (batches * genlen x tokens)
        dist = Categorical(logits = logits / temp)

        y = x[:, seed_offset:].reshape((batches * gen_len)) # single vector

        logpas = dist.log_prob(y).reshape((batches, gen_len))
        entropies = dist.entropy().reshape((batches, gen_len))
        return logpas, entropies

    def _compute_probabilities_SINGLE(self, x, context=None, temp=1.0):
        """Like the above, but only returns the value for the final token"""
#        batches, total_len = x.size()
#        gen_len = total_len - seed_offset
        # I think this is finally it. All the non-terminal tokens are input; the terminal one is index.
        logits = self.sentence_autoencoder(x[:, :-1], context=context, use_masks=True, return_full=False) # should be batches x tokens
        dist = Categorical(logits = logits / temp)

        inds = x[:, -1]

        logpas = dist.log_prob(inds)
        entropies = dist.entropy()
        return logpas, entropies

    # Experimental use of self.generate for a particular type of input question
    def answer_image_comparison_question(self, text_seed, img1_batch, img2_batch, maxlen = None, temp=1.0, ret_all=True, temp_eps = 1e-4):
        return self.generate(text_seed, self.two_img_context(img1_batch, img2_batch), maxlen, temp, ret_all, temp_eps)


