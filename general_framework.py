# This file should have all the code shared between all (or most) of the tasks
# should have all the torch libraries I need
# Updated for QwenBastardBrain with Qwen tokenizer and ProcessBench dataset

import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from visual_transformer import *
from visual_transformer.qwen_player import QwenBastardBrain

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from game import *

device = torch.device('cuda:0') # CHANGE THIS EVERY TIME
#device = torch.device('cuda:1') # CHANGE THIS EVERY TIME

########
# Model initialization from Qwen source
# Comment out this block when loading model from disk instead
########

# --- BEGIN MODEL INIT FROM QWEN ---
print("Loading Qwen/Qwen3-0.6B source model...")
_qwen_source = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    torch_dtype=torch.bfloat16
).to(device)

print("Creating QwenBastardBrain and stripping source model...")
model = QwenBastardBrain()
model.strip_source_model(_qwen_source)
model = model.to(device)

# Free the source model memory
del _qwen_source
torch.cuda.empty_cache()
print("Model ready!")
# --- END MODEL INIT FROM QWEN ---

# Alternative: Load from disk (uncomment when needed)
# model = QwenBastardBrain.from_pretrained("path/to/saved/model").to(device)
# --- or ---
# model = QwenBastardBrain()
# model.load_state_dict(torch.load("brain_checkpoints/model.pt"))
# model = model.to(device)

########
# Game setup
########

game_settings = BIG_tool_use_advanced_2_5
game_settings.gameSize = 224 # for compatibility with brain's expected size
G = discreteGame(game_settings)

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
        
        src_files = Path("./text_pretraining_data/").glob("*-eval.txt") if evaluate else Path("./text_pretraining_data/").glob("*-train.txt")
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

img_criterion = nn.MSELoss()

########
# Game utilities
########

def get_settings_batch(batch_size):
    return [G.random_bare_settings(gameSize=224, max_agent_offset=0.5) for i in range(batch_size)]

def get_images(settings_batch, device=device, ignore_agent=False, ignore_gold=False, ignore_walls=False):
    batch_size = len(settings_batch)
    img = torch.zeros(batch_size, 224, 224, 3)
    should_draw = (ignore_agent or ignore_gold or ignore_walls)
    for i in range(batch_size):
        G2 = discreteGame(settings_batch[i])
        if should_draw:
            G2.draw(ignore_agent=ignore_agent, ignore_gold=ignore_gold, ignore_wals=ignore_walls)
        img[i] = torch.tensor(G2.getData())
    img = torch.permute(img, (0, 3, 1, 2)).contiguous().to(device)
    return img
