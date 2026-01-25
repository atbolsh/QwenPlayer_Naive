# This file should have all the code shared between all (or most) of the tasks
# should have all the torch libraries I need
# Updated for QwenAgentPlayer with Qwen tokenizer

import json
import random
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visual_transformer import QwenAgentPlayer, QwenAgentPipe, QwenExtension
from visual_transformer.model import ImageTransformerEncoder, ImageTransformerDecoder

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from game import *

# Import lightweight game utilities (device, G, get_settings_batch, get_images, img_criterion)
from .general_framework_lightweight import device, G, game_settings, get_settings_batch, get_images, img_criterion

########
# Model initialization for QwenAgentPlayer
########

def create_model(model_name: str = "Qwen/Qwen3-0.6B", device=None, use_lora: bool = False):
    """
    Create a QwenAgentPlayer model.
    
    Args:
        model_name: Qwen model name/path
        device: Device to load model on
        use_lora: If True, wrap model with LoRA adapters for efficient training
        
    Returns:
        QwenAgentPlayer instance (optionally with LoRA)
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"Creating QwenAgentPlayer with {model_name}...")
    model = QwenAgentPlayer(
        model_name=model_name,
        embed_dim=1024,
        num_heads=8,
        device=device,
    )
    print("Model ready!")
    
    if use_lora:
        model = apply_lora_to_text(model)
    
    return model


def apply_lora_to_text(model, r=4, lora_alpha=16, lora_dropout=0.1):
    """
    Apply LoRA adapters to the text model (Qwen) for efficient training.
    
    Note: This only applies LoRA to the Qwen language model. The image
    encoder (img_enc) and decoder (img_dec) remain fully trainable.
    
    Args:
        model: QwenAgentPlayer instance
        r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: Dropout rate for LoRA layers
        
    Returns:
        Model with LoRA adapters applied to text model
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise ImportError("Please install peft: pip install peft")
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )
    
    # Apply LoRA to the underlying Qwen model
    model.pipe.model.qwen_model = get_peft_model(model.pipe.model.qwen_model, lora_config)
    print("LoRA adapters applied!")
    model.pipe.model.qwen_model.print_trainable_parameters()
    
    return model


# Create default model instance
print("Loading QwenAgentPlayer...")

# OLD METHOD: Create fresh model from HuggingFace
# model = create_model(device=device)

# NEW METHOD: Create model and load frankenstein checkpoint (pretrained vision encoder/decoder)
FRANKENSTEIN_CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "brain_checkpoints", "first_frankenstein.pt")
model = create_model(device=device)
if os.path.exists(FRANKENSTEIN_CHECKPOINT):
    print(f"Loading frankenstein checkpoint from {FRANKENSTEIN_CHECKPOINT}...")
    model.pipe.model.load_state_dict(torch.load(FRANKENSTEIN_CHECKPOINT, map_location=device))
    print("Frankenstein checkpoint loaded!")
else:
    print(f"WARNING: Frankenstein checkpoint not found at {FRANKENSTEIN_CHECKPOINT}")
    print("Using fresh model weights. Run frankensteinify.py first to create the checkpoint.")

########
# Qwen Tokenizer setup
########

vocab_size = 151936  # Qwen vocab size
model_name = "Qwen/Qwen3-0.6B"

# Load the Qwen tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Special tokens for game controls
# Note: Qwen tokenizer handles special tokens differently than ByteLevelBPE
# We add custom tokens for game control
SPECIAL_TOKENS = ['<forward>', '<clock>', '<anticlock>']
tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})

# Max sequence length for text
MAX_SEQ_LENGTH = 32

def encode_text(text, max_length=MAX_SEQ_LENGTH):
    """Encode text using Qwen tokenizer, returns tensor of token ids."""
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return encoded['input_ids'].squeeze(0)

def encode_batch(text_list, max_length=MAX_SEQ_LENGTH):
    """Encode a batch of texts using Qwen tokenizer."""
    encoded = tokenizer(
        text_list,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return encoded['input_ids']

def decode_text(token_ids, skip_special_tokens=False):
    """Decode token ids back to text."""
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

def decode_batch(token_ids_batch, skip_special_tokens=False):
    """Decode a batch of token ids back to text."""
    return tokenizer.batch_decode(token_ids_batch, skip_special_tokens=skip_special_tokens)

########
# Adapter functions for QwenAgentPlayer
# These provide backward compatibility with old EnhancedAgentBrain/QwenBastardBrain API
########

def model_forward(model, text_batch, img_batch, ret_imgs=True, generate_image=True):
    """
    Forward pass for string text inputs.
    
    Use model_forward_with_tokens() for token tensor inputs (more efficient).
    
    Args:
        model: QwenAgentPlayer instance
        text_batch: List of strings (NOT token tensors - use model_forward_with_tokens for that)
        img_batch: Image tensor (batch_size, 3, 224, 224)
        ret_imgs: Whether to return reconstructed images
        generate_image: Whether to generate output images
        
    Returns:
        If ret_imgs: (text_probs, img_recon)
        Otherwise: text_probs
        
        text_probs is in format (batch, vocab, seq_len) for backward compatibility
    """
    # This function is for string inputs - tokenize them first for efficiency
    if isinstance(text_batch, torch.Tensor):
        # If given tokens, just use model_forward_with_tokens directly
        return model_forward_with_tokens(model, text_batch, img_batch, ret_imgs)
    
    # Tokenize the strings
    texts = text_batch if isinstance(text_batch, list) else [text_batch]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    
    # Handle single image for batch
    if img_batch.dim() == 3:
        img_batch = img_batch.unsqueeze(0)
    img_batch = img_batch.to(device)
    
    # Create attention mask
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    attention_mask = (input_ids != pad_token_id).long()
    
    # Use batch_forward with the tokenized inputs
    result = model.batch_forward(
        input_ids=input_ids,
        image=img_batch,
        attention_mask=attention_mask,
        generate_image=ret_imgs and generate_image,
    )
    
    # Extract logits and convert to old format (batch, vocab, seq_len)
    logits = result['outputs'].logits  # (batch, seq_len, vocab)
    text_probs = logits.permute(0, 2, 1)  # (batch, vocab, seq_len)
    
    if ret_imgs:
        img_recon = result.get('generated_images')
        return text_probs, img_recon
    return text_probs


def model_forward_with_tokens(model, text_batch, img_batch, ret_imgs=True):
    """
    Forward pass using tokenized input directly.
    
    This is the primary interface for frameworks that work with token tensors.
    Operates directly on token tensors without decode/encode round-trip.
    
    Args:
        model: QwenAgentPlayer instance
        text_batch: Token tensor (batch_size, seq_len) - must be a tensor
        img_batch: Image tensor (batch_size, 3, 224, 224)
        ret_imgs: Whether to return reconstructed images
        
    Returns:
        If ret_imgs: (text_probs, img_recon)
        Otherwise: text_probs
        
        text_probs is in format (batch, vocab, seq_len) for backward compatibility
    """
    # Ensure text_batch is a tensor on the right device
    if not isinstance(text_batch, torch.Tensor):
        raise ValueError("model_forward_with_tokens expects token tensors, not strings. Use encode_batch() first.")
    
    input_ids = text_batch.to(device)
    
    # Create attention mask (1 for non-pad tokens)
    # Qwen uses pad_token_id, default to 0 if not set
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    attention_mask = (input_ids != pad_token_id).long()
    
    # Ensure img_batch has correct shape
    if img_batch.dim() == 3:
        img_batch = img_batch.unsqueeze(0)
    img_batch = img_batch.to(device)
    
    # Use batch_forward directly with token tensors - no encode/decode overhead
    result = model.batch_forward(
        input_ids=input_ids,
        image=img_batch,
        attention_mask=attention_mask,
        generate_image=ret_imgs,
    )
    
    # Extract logits and convert to old format (batch, vocab, seq_len)
    logits = result['outputs'].logits  # (batch, seq_len, vocab)
    text_probs = logits.permute(0, 2, 1)  # (batch, vocab, seq_len)
    
    if ret_imgs:
        img_recon = result.get('generated_images')
        return text_probs, img_recon
    return text_probs


########
# Dataset - using Qwen's ProcessBench
########

class ProcessBenchDataset(Dataset):
    """Dataset based on Qwen's ProcessBench for training."""
    def __init__(self, split='gsm8k', seq_length=MAX_SEQ_LENGTH, device=None):
        if device is None:
            device = 'cpu'
        self.device = device
        self.seq_length = seq_length
        
        # Load ProcessBench dataset
        self.dataset = load_dataset('Qwen/ProcessBench', split=split)
        
        # Pre-tokenize all examples
        self.examples = []
        for item in self.dataset:
            # ProcessBench has various fields - we'll use the main text content
            # Adjust field access based on actual dataset structure
            if 'problem' in item:
                text = item['problem']
            elif 'question' in item:
                text = item['question']
            elif 'text' in item:
                text = item['text']
            else:
                # Fallback: convert the whole item to string
                text = str(item)
            
            encoded = tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.seq_length,
                return_tensors='pt'
            )
            self.examples.append(encoded['input_ids'].squeeze(0))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return self.examples[i].to(self.device)


class SampleDataset(Dataset):
    """Legacy dataset for local text files, updated for Qwen tokenizer."""
    def __init__(self, seq_length=MAX_SEQ_LENGTH, evaluate=False, device=None):
        if device is None:
            device = 'cpu'
        self.device = device
        self.seq_length = seq_length
        
        self.examples = []
        
        # Look for data in parent directory
        data_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "text_pretraining_data"
        src_files = data_path.glob("*-eval.txt") if evaluate else data_path.glob("*-train.txt")
        for src_file in src_files:
            print("🔥", src_file)
            lines = src_file.read_text(encoding="utf-8").splitlines()
            for line in lines:
                if line.strip():  # Skip empty lines
                    encoded = tokenizer(
                        line,
                        padding='max_length',
                        truncation=True,
                        max_length=self.seq_length,
                        return_tensors='pt'
                    )
                    self.examples.append(encoded['input_ids'].squeeze(0))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return self.examples[i].to(self.device)


# Default datasets - use ProcessBench as primary
try:
    sdt = ProcessBenchDataset(split='gsm8k', device='cpu')
    sdv = ProcessBenchDataset(split='gsm8k', device='cpu')  # Same split for now; adjust as needed
    print(f"Loaded ProcessBench dataset with {len(sdt)} examples")
except Exception as e:
    print(f"Warning: Could not load ProcessBench dataset: {e}")
    print("Falling back to local SampleDataset")
    sdt = SampleDataset()
    sdv = SampleDataset(evaluate=True)

num_controls = len(sdt)

########
# Loss functions
########

ent_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

def get_text_loss(res, inputs):
    """Compute cross-entropy loss for text prediction."""
    return torch.sum(ent_criterion(res[:, :, :-1], inputs[:, 1:]))
