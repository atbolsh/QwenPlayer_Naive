# Extends the 'Player' framework by using a pretrained Qwen model as the language embeddings

from .enhanced_model import *
from .model import *
from .memory import *
from .vision_canvas import *

###########################

# Qwen imports:

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from transformers import AutoModelForCausalLM, AutoTokenizer

#custom addition
from visual_transformer import *
from typing import Callable, Optional, Union

####

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs

###########################

class Qwen3_BastardEncoder(
    nn.Module,
    PyTorchModelHubMixin, 
    # optionally, you can add metadata which gets pushed to the model card
    repo_url="qwen3-0.6G-encoder",
    pipeline_tag="text-to-vector", # look up correct term here
    license="mit",
):
    def __init__(self, source_model=None):
        super().__init__()
        if source_model:
            self.strip_source_model(source_model)
        else:
            self.num_layers = 14
            self.embed_tokens = None
            self.rotary_emb = None
            self.layers = None

    def strip_source_model(self, source_model):
        self.num_layers = source_model.model.config.num_hidden_layers // 2
        self.embed_tokens = source_model.model.embed_tokens
        self.rotary_emb = source_model.model.rotary_emb
        self.layers = nn.Sequential(*[source_model.model.layers[i] for i in range(self.num_layers)])
        return None

    def get_device(self):
        return self.rotary_emb._buffers['inv_freq'].device

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.FloatTensor] = None
    ):
        if (input_ids is None) ^ (input_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or input_embeds")

        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)

        position_ids = torch.arange(
            0, input_embeds.shape[1], device=input_embeds.device
        ).unsqueeze(0)

        #attention_mask = torch.ones(input_embeds.shape[:-1], dtype=input_embeds.dtype, device=input_embeds.device) #None # Paleolithic holdover

        hidden_states = input_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=None,#attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=None, # double-check this one
                position_embeddings=position_embeddings
            )

        return hidden_states



class Qwen3_BastardDecoder(
    nn.Module,
    PyTorchModelHubMixin, 
    # optionally, you can add metadata which gets pushed to the model card
    repo_url="qwen3-0.6G-decoder",
    pipeline_tag="text-to-vector", # look up correct term here
    license="mit",
):
    def __init__(self, source_model=None):
        super().__init__()
        if source_model:
            self.strip_source_model(source_model)
        else:
            self.num_layers = 14
            self.rotary_emb = None
            self.layers = None
            self.norm = None
            self.lm_head = None

    def strip_source_model(self, source_model):
        self.num_layers = source_model.model.config.num_hidden_layers // 2
        self.rotary_emb = source_model.model.rotary_emb
        self.layers = nn.Sequential(*[source_model.model.layers[i] for i in range(self.num_layers, self.num_layers*2)])
        self.norm = source_model.model.norm
        self.lm_head = source_model.lm_head
        return None

    def get_device(self):
        return self.rotary_emb._buffers['inv_freq'].device

    def forward(
        self,
        input_embeds: Optional[torch.FloatTensor] = None,
        context: Optional[torch.FloatTensor] = None
#        skip_initial: Optional[int] = 0 #64 # how many initial tokens to skip when return results
    ):
        if input_embeds is None:
            raise ValueError("You must provide input_embeds")

        if context is not None:
            input_embeds = torch.cat((context, input_embeds), dim=1)
            skip_initial = context.size()[1]
        else:
            skip_initial = 0

        position_ids = torch.arange(
            0, input_embeds.shape[1], device=input_embeds.device
        ).unsqueeze(0)

        #attention_mask = torch.ones(input_embeds.shape[:-1], dtype=input_embeds.dtype, device=input_embeds.device) #None # Paleolithic holdover

        hidden_states = input_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=None,#{'full_attention': None},#attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=None, # double-check this one
                position_embeddings=position_embeddings
            )
            # there may be weird copying bugs here; will work on it later
            if skip_initial:
                hidden_states[:, :skip_initial, :] = context

        hidden_states = self.norm(hidden_states[:, skip_initial:, :])
        out_logits = self.lm_head(hidden_states)

        return out_logits






