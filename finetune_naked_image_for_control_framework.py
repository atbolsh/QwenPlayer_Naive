"""Finetune: Naked image autoencoder for control framework in Float32

This script finetunes the decoder to work with realistic Qwen3 hidden states
as context, simulating how it will be used in the control framework.

Key changes:
1. Loads finetune_control_step_001000.pt to initialize weights
2. Uses G.random_full_image_set() - same images as control framework
3. Decoder receives sampled Qwen3 hidden states as context (magnitude ~100)
4. Embedding tensor refreshed periodically during training
"""

import os
import csv
import random
import gc
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image

# Load environment variables for HF token
from dotenv import load_dotenv
load_dotenv()

from transformers import AutoTokenizer, AutoModelForCausalLM
from visual_transformer.model import ImageTransformerEncoder, ImageTransformerDecoder
from general_framework_lightweight import device, img_criterion, G, load_text_datasets, get_text_batch

# Default dtype - using float32 for training stability
DEFAULT_DTYPE = torch.float32

# Directories
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "brain_checkpoints")
DEMO_DIR = os.path.join(os.path.dirname(__file__), "demo_images")
LEDGER_PATH = os.path.join(os.path.dirname(__file__), "finetune_control_better_embeddings_losses.csv")

# Checkpoint to load
INIT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "finetune_control_step_001000.pt")

# Save prefix
SAVE_PREFIX = "finetune_control_better_embeddings"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DEMO_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 1100
LEARNING_RATE = 1e-5
NUM_STEPS = 10000000
PRINT_EVERY = 100
SAVE_EVERY = 1000
REFRESH_EMBEDDINGS_EVERY = 100  # Refresh embedding tensor every N steps

# Stability settings
GRADIENT_CLIP_VALUE = 1.0

# Embed dim
EMBED_DIM = 1024
CONTEXT_SEQ_LEN = 32
NUM_EMBEDDING_LINES = 100  # Number of text lines to embed

# Global embedding tensor (will be initialized later)
EMBEDDING_TENSOR = None
sdt = None  # Text dataset


def initialize_embedding_tensor(num_lines=NUM_EMBEDDING_LINES, seq_len=CONTEXT_SEQ_LEN, 
                                 target_device=device, dtype=DEFAULT_DTYPE):
    """
    Initialize a tensor of Qwen3 hidden states from random text lines.
    
    Loads Qwen3, embeds text, extracts hidden states, then unloads the model.
    
    Returns:
        Tensor of shape (num_lines, seq_len, embed_dim) containing detached hidden states
    """
    global sdt
    
    print(f"Initializing embedding tensor ({num_lines} lines, {seq_len} tokens each)...")
    
    # Load text dataset if not already loaded
    if sdt is None:
        print("Loading text dataset...")
        sdt, _ = load_text_datasets()
    
    # Load Qwen3 model temporarily
    print("Loading Qwen3-0.6B for embedding extraction...")
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    qwen_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.bfloat16,  # Use bf16 for efficiency
        tie_word_embeddings=False
    ).to(target_device)
    qwen_model.eval()
    
    # Get random text batches and extract hidden states
    embeddings_list = []
    
    with torch.no_grad():
        # Process in smaller batches to avoid OOM
        batch_size = 20
        for batch_start in range(0, num_lines, batch_size):
            batch_end = min(batch_start + batch_size, num_lines)
            current_batch_size = batch_end - batch_start
            
            # Get random starting indices
            start_idx = random.randint(0, len(sdt) - current_batch_size - 1)
            text_batch = get_text_batch(sdt, start_idx, current_batch_size)
            
            # Truncate/pad to seq_len
            if text_batch.shape[1] > seq_len:
                text_batch = text_batch[:, :seq_len]
            elif text_batch.shape[1] < seq_len:
                # Pad with zeros (will still produce embeddings)
                padding = torch.zeros(current_batch_size, seq_len - text_batch.shape[1], 
                                     dtype=text_batch.dtype, device=text_batch.device)
                text_batch = torch.cat([text_batch, padding], dim=1)
            
            text_batch = text_batch.to(target_device)
            
            # Forward pass to get hidden states
            outputs = qwen_model(text_batch, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last layer: (batch, seq_len, 1024)
            
            # Convert to target dtype and detach
            hidden_states = hidden_states.to(dtype).detach()
            embeddings_list.append(hidden_states)
    
    # Concatenate all batches
    embedding_tensor = torch.cat(embeddings_list, dim=0)  # (num_lines, seq_len, embed_dim)
    
    print(f"Embedding tensor shape: {embedding_tensor.shape}")
    print(f"Embedding magnitude (per token): {torch.norm(embedding_tensor, dim=2).mean().item():.2f}")
    
    # Unload Qwen3 to free VRAM
    print("Unloading Qwen3 model...")
    del qwen_model
    del qwen_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("Qwen3 unloaded, VRAM freed.")
    
    return embedding_tensor


def sample_context_from_embeddings(embedding_tensor, batch_size, seq_len=CONTEXT_SEQ_LEN):
    """
    Sample random context vectors from the embedding tensor.
    
    Samples random tokens from random lines - doesn't need to be coherent.
    
    Args:
        embedding_tensor: (num_lines, seq_len, embed_dim)
        batch_size: Number of context sequences to generate
        seq_len: Length of each context sequence
        
    Returns:
        Tensor of shape (batch_size, seq_len, embed_dim)
    """
    num_lines, embed_seq_len, embed_dim = embedding_tensor.shape
    
    # For each batch item, sample seq_len random tokens from random lines
    # This creates diverse, realistic-magnitude context
    contexts = []
    for _ in range(batch_size):
        # Sample random line indices and token indices
        line_indices = torch.randint(0, num_lines, (seq_len,))
        token_indices = torch.randint(0, embed_seq_len, (seq_len,))
        
        # Gather the tokens
        context = embedding_tensor[line_indices, token_indices, :]  # (seq_len, embed_dim)
        contexts.append(context)
    
    return torch.stack(contexts, dim=0)  # (batch_size, seq_len, embed_dim)


class NakedImageAutoencoderWithRealisticContext(nn.Module):
    """Standalone image autoencoder with realistic Qwen3 hidden states as context.
    
    The decoder receives sampled hidden states from Qwen3, matching the magnitude
    and distribution it will see in the control framework.
    """
    
    def __init__(self, embed_dim=1024, num_heads=8, dtype=DEFAULT_DTYPE, context_seq_len=32):
        super().__init__()
        self.dtype = dtype
        self.embed_dim = embed_dim
        self.context_seq_len = context_seq_len
        self.img_enc = ImageTransformerEncoder(embed_dim=embed_dim, num_heads=num_heads, dtype=dtype)
        self.img_dec = ImageTransformerDecoder(embed_dim=embed_dim, num_heads=num_heads, dtype=dtype)
    
    def forward(self, img_batch, context):
        """
        Forward pass with provided context.
        
        Args:
            img_batch: Images (batch_size, 3, 224, 224)
            context: Pre-sampled context (batch_size, context_seq_len, embed_dim)
        """
        # Encode the image
        encoding = self.img_enc(img_batch)
        
        # Decode with provided context
        reconstruction = self.img_dec(encoding, context)
        return reconstruction
    
    def get_device(self):
        return self.img_enc.get_device()


def get_control_images(batch_size, device, dtype=DEFAULT_DTYPE):
    """Get images using the same method as control framework."""
    img_set = G.random_full_image_set(restrict_angles=True)
    np_b_size = img_set.shape[0]
    
    if batch_size < np_b_size:
        inds = random.sample(list(range(np_b_size)), batch_size)
        img_set = img_set[inds]
    
    if batch_size > np_b_size:
        while img_set.shape[0] < batch_size:
            img_set = np.concatenate((img_set, G.random_full_image_set(restrict_angles=True)))
        if img_set.shape[0] > batch_size:
            inds = random.sample(list(range(img_set.shape[0])), batch_size)
            img_set = img_set[inds]
    
    img_tensor = torch.permute(torch.tensor(img_set, dtype=dtype).to(device), (0, 3, 1, 2))
    return img_tensor


def save_demo_image(model, embedding_tensor, step, device):
    """Save a single input-output pair."""
    model.eval()
    with torch.no_grad():
        img = get_control_images(1, device=device, dtype=DEFAULT_DTYPE)
        context = sample_context_from_embeddings(embedding_tensor, 1, CONTEXT_SEQ_LEN)
        recon = model(img, context)
        
        input_path = os.path.join(DEMO_DIR, f"{SAVE_PREFIX}_step_{step:06d}_input.png")
        output_path = os.path.join(DEMO_DIR, f"{SAVE_PREFIX}_step_{step:06d}_output.png")
        save_image(img[0], input_path)
        save_image(recon[0].clamp(0, 1), output_path)
    model.train()


# ============== MAIN TRAINING ==============

# Initialize embedding tensor (loads and unloads Qwen3)
EMBEDDING_TENSOR = initialize_embedding_tensor()

# Initialize CSV ledger
with open(LEDGER_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'mse_loss'])

# Initialize model (float32)
print("Initializing NakedImageAutoencoderWithRealisticContext (float32)...")
model = NakedImageAutoencoderWithRealisticContext(
    embed_dim=EMBED_DIM, 
    num_heads=8, 
    dtype=DEFAULT_DTYPE,
    context_seq_len=CONTEXT_SEQ_LEN
).to(device)

# Load checkpoint
if os.path.exists(INIT_CHECKPOINT):
    print(f"Loading checkpoint from {INIT_CHECKPOINT}...")
    state_dict = torch.load(INIT_CHECKPOINT, map_location=device, weights_only=True)
    
    # Filter to only load img_enc and img_dec weights
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('img_enc.') or key.startswith('img_dec.'):
            filtered_state_dict[key] = value
    
    model.load_state_dict(filtered_state_dict, strict=False)
    print("Checkpoint loaded (img_enc and img_dec weights)!")
else:
    print(f"ERROR: Checkpoint not found at {INIT_CHECKPOINT}")
    print("Please ensure the checkpoint file exists.")
    exit(1)

model.train()

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Loss function
criterion = nn.MSELoss()

print(f"Finetuning image autoencoder for control framework (float32) for {NUM_STEPS} steps...")
print(f"Using realistic Qwen3 hidden states as context (magnitude ~100)")
print(f"Context shape: batch x {CONTEXT_SEQ_LEN} x {EMBED_DIM}")
print(f"Embedding tensor refreshed every {REFRESH_EMBEDDINGS_EVERY} steps")
print(f"Checkpoints saved every {SAVE_EVERY} steps")
print(f"Losses logged to {LEDGER_PATH}")
print(f"Gradient clip value: {GRADIENT_CLIP_VALUE}")

for step in range(NUM_STEPS):
    # Refresh embedding tensor periodically
    if step > 0 and step % REFRESH_EMBEDDINGS_EVERY == 0:
        print(f"Refreshing embedding tensor at step {step}...")
        EMBEDDING_TENSOR = initialize_embedding_tensor()
    
    # Generate game images (float32) - same as control framework
    img_batch = get_control_images(BATCH_SIZE, device=device, dtype=DEFAULT_DTYPE)
    
    # Sample context from embedding tensor
    context = sample_context_from_embeddings(EMBEDDING_TENSOR, BATCH_SIZE, CONTEXT_SEQ_LEN)
    
    # Forward pass
    reconstructed = model(img_batch, context)
    
    # Compute loss
    loss = criterion(reconstructed, img_batch)
    
    # Check for nan/inf
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"WARNING: Loss is {loss.item()} at step {step+1}, skipping...")
        optimizer.zero_grad()
        continue
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Clip gradients to prevent explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
    
    # Optimizer step
    optimizer.step()
    
    if ((step + 1) % PRINT_EVERY == 0) or (step < 10):
        loss_val = loss.item()
        print(f"Step {step+1}/{NUM_STEPS} | MSE Loss: {loss_val:.6f}")
        
        # Log to CSV
        with open(LEDGER_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step + 1, loss_val])
    
    # Save checkpoint and demo image every SAVE_EVERY steps
    if (step + 1) % SAVE_EVERY == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{SAVE_PREFIX}_step_{step+1:06d}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        save_demo_image(model, EMBEDDING_TENSOR, step + 1, device)
        print(f"Demo image saved for step {step+1}")

print("Training complete!")

# Final eval
model.eval()
with torch.no_grad():
    test_imgs = get_control_images(4, device=device, dtype=DEFAULT_DTYPE)
    test_context = sample_context_from_embeddings(EMBEDDING_TENSOR, 4, CONTEXT_SEQ_LEN)
    
    test_recon = model(test_imgs, test_context)
    test_loss = criterion(test_recon, test_imgs)
    
    print(f"Final eval MSE: {test_loss.item():.6f}")

# Save final checkpoint
final_path = os.path.join(CHECKPOINT_DIR, f"{SAVE_PREFIX}_final.pt")
torch.save(model.state_dict(), final_path)
print(f"Final finetuned model saved to {final_path}")
