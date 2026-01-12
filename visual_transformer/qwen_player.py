# QwenBastardBrain - A multimodal neural network using Qwen3 encoder/decoder
# Near-identical copy of EnhancedAgentBrain with Qwen-based text processing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import math

from huggingface_hub import PyTorchModelHubMixin

# Import from other files in visual_transformer
from .model import (
    ImageTransformerEncoder,
    ImageTransformerDecoder,
    generate_src_mask,
    generate_src_padding_mask,
)
from .enhanced_model import DopamineWrapper, VisionWeightedSum
from .memory import Memory, MemoryEncoder
from .vision_canvas import VisionCanvases
from .qwen_encoders import Qwen3_BastardEncoder, Qwen3_BastardDecoder


class QwenBastardBrain(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="qwen-bastard-brain",
    pipeline_tag="multimodal",
    license="mit",
):
    def __init__(self, vocab_size=151936, mem_size=128, new_tokens=1):
        super().__init__()
        
        # All components use 1024 embed_dim consistently
        embed_dim = 1024
        
        # Image encoder/decoder with 1024 dim
        self.img_enc = ImageTransformerEncoder(embed_dim=embed_dim)
        self.img_dec = ImageTransformerDecoder(embed_dim=embed_dim)
        self.img_weight = VisionWeightedSum(embed_dim=embed_dim)

        # Qwen-based text encoder/decoder (1024 dim, bfloat16)
        self.text_enc = Qwen3_BastardEncoder()
        self.text_dec = Qwen3_BastardDecoder()
        
        # Store vocab_size for use in compute_probabilities
        self.vocab_size = vocab_size

        # Dopamine for RL - 1024 dim
        self.dopamine = DopamineWrapper(embed_dim=embed_dim)

        # 7 inputs at 1024 dim: 3 img canvases, 1 img input, memory, dopamine, text input
        self.context_tagging = nn.Parameter(torch.empty((7, 1, embed_dim)))
        nn.init.uniform_(self.context_tagging, -1.0/math.sqrt(embed_dim), 1.0/math.sqrt(embed_dim))

        # Memory processing with 1024 dim
        self.memory = Memory(mem_size, new_tokens, vector_dim=embed_dim)
        self.mem_enc = MemoryEncoder(new_tokens=new_tokens, embed_dim=embed_dim)

        # Store embed_dim for reference
        self.embed_dim = embed_dim

        # set image canvases and None for self.context
        self.reset()

    def reset(self):
        # note this may create orphaned tensors and memory leaks down the line.
        self.canvases = VisionCanvases(3)
        self.canvases.to(self.get_device())
        self.memory = Memory(self.memory.mem_size, self.memory.new_tokens, vector_dim=self.embed_dim)
        self.memory.to(self.get_device())
        self.context = None

    def soft_reset(self):
        self.memory.soft_reset()
        self.canvases.soft_reset()

    def move_to(self, device):
        self.canvases.to(device)
        return self.to(device)

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

    def get_text_encoding(self, text_batch, src_attention_mask=None, src_key_padding_mask=None):
        # Qwen encoder ignores masks (uses its own attention mechanism)
        # Output is bfloat16, 1024 dim - convert to float32 for consistency
        encoding = self.text_enc(input_ids=text_batch)
        return encoding.float()

    def get_text_decoding(self, text_encoding, src_attention_mask=None, src_key_padding_mask=None, context=None, return_full=True):
        # Convert inputs to bfloat16 for Qwen decoder
        text_encoding_bf16 = text_encoding.to(torch.bfloat16)
        
        if context is not None:
            context_bf16 = context.to(torch.bfloat16)
        else:
            context_bf16 = None
        
        # Get logits from Qwen decoder
        logits = self.text_dec(input_embeds=text_encoding_bf16, context=context_bf16)
        
        # Convert output back to float32 for compatibility
        logits = logits.float()
        
        if return_full:
            # Return logits in format: batch x vocab x seq_len (to match original)
            return logits.permute(0, 2, 1)
        else:
            # Only last token
            return logits[:, -1, :]

    def forward(self, text_batch, img_batch=None, ret_imgs=False, return_full=True, use_masks=True, create_context=True, ret_dopamine=False, ret_img_weight=False):
        if (img_batch is None) and create_context:
            raise ValueError("Must provide img_batch to create new context")
        if ret_imgs and (not create_context):
            raise ValueError("to generate new images, create_context must be true")
        if ret_dopamine and (not ret_imgs):
            raise ValueError("ret_dopamine is just an extension of ret_imgs; please set that flag, too")
        if ret_dopamine and ret_img_weight:
            raise NotImplementedError("Haven't made this fix; may need a dict-style return class instead")

        b = text_batch.size()[0]
        src_attention_mask, src_key_padding_mask = self.get_masks(text_batch, use_masks)
        
        # Get text encoding (converted to float32, 1024 dim)
        text_encoding = self.get_text_encoding(text_batch, src_attention_mask, src_key_padding_mask)

        if create_context:
            if self.canvases.is_empty():
                self.canvases.store(img_batch)

            context = []

            real_img_context = self.img_enc(img_batch)  # 1024 dim, float32
            context.append(real_img_context)

            for i in range(self.canvases.num_canvases):
                context.append(self.img_enc(self.canvases[i]))

            if self.memory.is_empty():
                context.append(torch.zeros(b, 128, self.embed_dim, device=text_batch.device))
            else:
                context.append(self.memory.memory)

            for i in range(len(context)):
                context[i] += self.context_tagging[i]
            tensor_context = torch.cat(context, dim=1)

            # Dopamine operates on 1024 dim features
            reaction = self.dopamine(real_img_context, tensor_context)
            context.append(reaction + self.context_tagging[-2])  # -1 will be for text input
            tensor_context = torch.cat((tensor_context, context[-1]), dim=1)

            self.context = tensor_context

        # Get text probabilities
        text_probs = self.get_text_decoding(text_encoding, src_attention_mask, src_key_padding_mask, self.context, return_full)
        
        # Store in memory
        self.memory.remember(self.mem_enc(text_encoding, self.context))

        if create_context:
            # For images and memory, add text_encoding to context
            context.append(text_encoding + self.context_tagging[-1])
            full_context = torch.cat((tensor_context, context[-1]), dim=1)

            # Image weight selection
            num_imgs = self.canvases.num_canvases + 1
            img_weights = self.img_weight(full_context)  # b x 4 x 1
            all_img_features = torch.cat([t.unsqueeze(1) for t in context[:num_imgs]], dim=1)  # b x 4 x 256 x 1024
            input_img_features = (all_img_features * img_weights.view(b, num_imgs, 1, 1)).sum(dim=1)  # b x 256 x 1024

            # Image reconstruction
            img_recon = self.img_dec(input_img_features, full_context)
            self.canvases.store(img_recon)

        if ret_imgs:
            if ret_dopamine:
                return text_probs, img_recon, reaction[:, 0, 0]
            elif ret_img_weight:
                return text_probs, img_recon, img_weights
            else:
                return text_probs, img_recon
        else:
            return text_probs

    def old_forward(self, text_batch, img_batch=None, ret_imgs=False, return_full=True, use_masks=True):
        src_attention_mask, src_key_padding_mask = self.get_masks(text_batch, use_masks)
        text_encoding = self.get_text_encoding(text_batch, src_attention_mask, src_key_padding_mask)
        
        if img_batch is None:
            img_context = text_encoding
        else:
            img_context = self.img_enc(img_batch)
        
        text_probs = self.get_text_decoding(text_encoding, src_attention_mask, src_key_padding_mask, img_context, return_full)
        
        if not ret_imgs:
            return text_probs
        else:
            if img_batch is None:
                batches = text_batch.size()[0]
                img_encoding = torch.zeros((batches, 256, self.embed_dim), device=self.get_device())
            else:
                img_encoding = img_context
            img_reconstruction = self.img_dec(img_encoding, text_encoding)
            return text_probs, img_reconstruction

    def qa_forward(self, text_batch_in, text_batch_out, img_batch=None, ret_imgs=False, return_full=True, use_masks=True):
        src_attention_mask_in, src_key_padding_mask_in = self.get_masks(text_batch_in, use_masks)
        src_attention_mask_out, src_key_padding_mask_out = self.get_masks(text_batch_out, use_masks)
        text_encoding_in = self.get_text_encoding(text_batch_in, src_attention_mask_in, src_key_padding_mask_in)
        text_encoding_out = self.get_text_encoding(text_batch_out, src_attention_mask_out, src_key_padding_mask_out)
        
        if img_batch is None:
            img_context = text_encoding_in
        else:
            img_context = self.img_enc(img_batch)
        
        text_probs = self.get_text_decoding(text_encoding_out, src_attention_mask_out, src_key_padding_mask_out, img_context, return_full)
        
        if not ret_imgs:
            return text_probs
        else:
            if img_batch is None:
                batches = text_batch_in.size()[0]
                img_encoding = torch.zeros((batches, 256, self.embed_dim), device=self.get_device())
            else:
                img_encoding = img_context
            img_reconstruction = self.img_dec(img_encoding, text_encoding_in)
            return text_probs, img_reconstruction

    def img_autoencoder(self, img_batch, context=None):
        img_encoding = self.img_enc(img_batch)
        if context is None:
            context = img_encoding
        return self.img_dec(img_encoding, context)

    def sentence_autoencoder(self, text_batch, context=None, return_full=True, use_masks=False, store_memory=False):
        src_attention_mask, src_key_padding_mask = self.get_masks(text_batch, use_masks)
        text_encoding = self.get_text_encoding(text_batch, src_attention_mask, src_key_padding_mask)
        
        if store_memory:
            self.memory.remember(self.mem_enc(text_encoding, context))
        
        return self.get_text_decoding(text_encoding, src_attention_mask, src_key_padding_mask, context, return_full)

    def evaluate_text(self, text_batch, img_batch=None, img_gradient=True, text_gradient=True):
        if text_gradient:
            text_encoding = self.text_enc(input_ids=text_batch).float()
        else:
            with torch.no_grad():
                text_encoding = self.text_enc(input_ids=text_batch).float()
        
        if img_batch is None:
            return self.dopamine(text_encoding)
        else:
            if img_gradient:
                context = self.img_enc(img_batch)
            else:
                with torch.no_grad():
                    context = self.img_enc(img_batch)
            return self.dopamine(text_encoding, context)

    def select(self, logits, temp=0.0, ret_all=True, temp_eps=1e-4):
        if temp < temp_eps:
            preds = torch.argmax(logits, dim=1)
            if not ret_all:
                return preds
            log_probs = torch.max(F.log_softmax(logits, dim=1), dim=1)
            dist = Categorical(logits=logits)
            entropy = dist.entropy()
            return preds, log_probs, entropy
        else:
            dist = Categorical(logits=logits / temp)
            preds = dist.sample()
            if not ret_all:
                return preds
            entropy = dist.entropy()
            log_probs = dist.log_prob(preds)
            return preds, log_probs, entropy

    def extend(self, seed, is_terminated, context=None, temp=1.0, ret_all=True, temp_eps=1e-4, store_memory=True):
        s = seed.size()
        output = torch.zeros((s[0], s[1] + 1), dtype=torch.long, device=seed.device)
        output[:, :-1] += seed
        logits = self.sentence_autoencoder(output, context, use_masks=True, return_full=False, store_memory=store_memory)
        if not ret_all:
            preds = self.select(logits, temp, ret_all, temp_eps)
        else:
            preds, log_probs, entropy = self.select(logits, temp, ret_all, temp_eps)
        preds = preds * torch.logical_not(is_terminated)
        output[:, -1] += preds
        is_terminated = torch.logical_or(is_terminated, (preds == 2))
        if not ret_all:
            return output, preds, is_terminated
        else:
            return output, preds, log_probs, entropy, is_terminated

    def generate(self, x=None, context=None, maxlen=None, temp=1.0, ret_all=True, temp_eps=1e-4, default_batches=1, store_memory=True):
        if maxlen is None:
            # Default to a reasonable sequence length for Qwen
            maxlen = 32
        if x is None:
            x = torch.zeros((default_batches, 1), device=self.get_device(), dtype=torch.long)
        if ret_all:
            lp = torch.zeros((default_batches, 1), device=self.get_device())
            ent = torch.zeros((default_batches, 1), device=self.get_device())
        batches, _ = x.size()
        is_terminated = torch.zeros(batches, dtype=torch.bool, device=self.get_device())
        if ret_all:
            lp = torch.zeros((batches, 1), device=self.get_device())
            ent = torch.zeros((batches, 1), device=self.get_device())
        firstGone = False
        while (x.size()[1] < maxlen) and (not torch.all(is_terminated)):
            if ret_all:
                x, _, newlp, newent, is_terminated = self.extend(x, is_terminated, context, temp, ret_all, temp_eps, store_memory)
                if firstGone:
                    lp = F.pad(lp, (0, 1))
                    ent = F.pad(ent, (0, 1))
                else:
                    firstGone = True
                lp[:, -1] += newlp
                ent[:, -1] += newent
            else:
                x, _, is_terminated = self.extend(x, is_terminated, context, temp, ret_all, temp_eps, store_memory)
        if ret_all:
            return x, lp, ent
        else:
            return x

    def compute_probabilities(self, x, seed_offset=1, context=None, temp=1.0, single=False, store_memory=False):
        if single:
            return self._compute_probabilities_SINGLE(x, context, temp, store_memory)
        else:
            return self._compute_probabilities_MULTI(x, seed_offset, context, temp, store_memory)

    def _compute_probabilities_MULTI(self, x, seed_offset, context=None, temp=1.0, store_memory=False):
        """Given sentences x, possibly computed by another model, compute the logpas and entropies for all the values chosen."""
        batches, total_len = x.size()
        gen_len = total_len - seed_offset
        logits = self.sentence_autoencoder(x, context=context, use_masks=True, return_full=True, store_memory=store_memory)[:, :, seed_offset-1:-1]
        logits = logits.transpose(1, 2)
        logits = logits.reshape((batches * gen_len, self.vocab_size))
        dist = Categorical(logits=logits / temp)

        y = x[:, seed_offset:].reshape((batches * gen_len))

        logpas = dist.log_prob(y).reshape((batches, gen_len))
        entropies = dist.entropy().reshape((batches, gen_len))
        return logpas, entropies

    def _compute_probabilities_SINGLE(self, x, context=None, temp=1.0, store_memory=False):
        """Like the above, but only returns the value for the final token"""
        logits = self.sentence_autoencoder(x[:, :-1], context=context, use_masks=True, return_full=False, store_memory=store_memory)
        dist = Categorical(logits=logits / temp)

        inds = x[:, -1]

        logpas = dist.log_prob(inds)
        entropies = dist.entropy()
        return logpas, entropies
