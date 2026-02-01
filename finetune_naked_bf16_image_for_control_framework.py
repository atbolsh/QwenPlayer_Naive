"""Finetune: Naked image autoencoder for control framework in BFloat16

This script finetunes the decoder to work with random noise as context
instead of the image encoding, simulating how it will be used in the 
control framework where text embeddings are passed as context.

Key changes from train_naked_image_bf16.py:
1. Loads frankenstein_bf16.pt to initialize weights
2. Uses G.random_full_image_set() - same images as control framework
3. Decoder receives random noise as context instead of image encoding

Note: ImageTransformerEncoder/Decoder now default to bf16, so no explicit conversion needed.
"""

import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image

from visual_transformer.model import ImageTransformerEncoder, ImageTransformerDecoder
from general_framework_lightweight import device, img_criterion, G

# Default dtype
DEFAULT_DTYPE = torch.bfloat16

# Directories
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "brain_checkpoints")
DEMO_DIR = os.path.join(os.path.dirname(__file__), "demo_images")
LEDGER_PATH = os.path.join(os.path.dirname(__file__), "finetune_control_losses.csv")

# Checkpoint to load (required - loads frankenstein_bf16.pt)
FRANKENSTEIN_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "frankenstein_bf16.pt")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DEMO_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 1100
LEARNING_RATE = 1e-5
NUM_STEPS = 10000000
PRINT_EVERY = 100
SAVE_EVERY = 1000

# bf16 stability settings
GRADIENT_CLIP_VALUE = 1.0  # Clip gradients to prevent explosion

# Embed dim (for random context generation)
EMBED_DIM = 1024
CONTEXT_SEQ_LEN = 32  # Simulate ~32 text tokens


class NakedImageAutoencoderWithRandomContext(nn.Module):
    """Standalone image autoencoder in bf16 with random context for decoder.
    
    The decoder receives random noise as context instead of the image encoding,
    simulating how it will be used in the control framework where text 
    embeddings (not image data) are passed as context.
    """
    
    def __init__(self, embed_dim=1024, num_heads=8, dtype=DEFAULT_DTYPE, context_seq_len=32):
        super().__init__()
        self.dtype = dtype
        self.embed_dim = embed_dim
        self.context_seq_len = context_seq_len
        # img_enc and img_dec now default to bf16
        self.img_enc = ImageTransformerEncoder(embed_dim=embed_dim, num_heads=num_heads, dtype=dtype)
        self.img_dec = ImageTransformerDecoder(embed_dim=embed_dim, num_heads=num_heads, dtype=dtype)
    
    def forward(self, img_batch):
        batch_size = img_batch.shape[0]
        device = img_batch.device
        
        # Encode the image
        encoding = self.img_enc(img_batch)
        
        # Generate random context that simulates text embeddings
        # Shape: (batch_size, context_seq_len, embed_dim)
        # Divide by sqrt(embed_dim) to get ~1.0 magnitude per feature
        random_context = torch.randn(
            batch_size, self.context_seq_len, self.embed_dim,
            device=device, dtype=self.dtype
        ) / 32.0  # 32.0 = sqrt(1024)
        
        # Decode with random context instead of image encoding
        reconstruction = self.img_dec(encoding, random_context)
        return reconstruction
    
    def get_device(self):
        return self.img_enc.get_device()


def get_control_images(batch_size, device, dtype=DEFAULT_DTYPE):
    """Get images using the same method as control framework."""
    # Use G.random_full_image_set() - same as control framework
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


def save_demo_image(model, step, device):
    """Save a single input-output pair."""
    model.eval()
    with torch.no_grad():
        # Use same image generation as control framework
        img = get_control_images(1, device=device, dtype=DEFAULT_DTYPE)
        recon = model(img)
        
        input_path = os.path.join(DEMO_DIR, f"finetune_control_step_{step:06d}_input.png")
        output_path = os.path.join(DEMO_DIR, f"finetune_control_step_{step:06d}_output.png")
        # Convert to float32 for saving (torchvision requires it)
        save_image(img[0].float(), input_path)
        save_image(recon[0].float().clamp(0, 1), output_path)
    model.train()


# Initialize CSV ledger
with open(LEDGER_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'mse_loss'])

# Initialize model (bf16 by default)
print("Initializing NakedImageAutoencoderWithRandomContext...")
model = NakedImageAutoencoderWithRandomContext(
    embed_dim=EMBED_DIM, 
    num_heads=8, 
    dtype=DEFAULT_DTYPE,
    context_seq_len=CONTEXT_SEQ_LEN
).to(device)

# Load frankenstein checkpoint
if os.path.exists(FRANKENSTEIN_CHECKPOINT):
    print(f"Loading frankenstein checkpoint from {FRANKENSTEIN_CHECKPOINT}...")
    state_dict = torch.load(FRANKENSTEIN_CHECKPOINT, map_location=device, weights_only=True)
    
    # Filter to only load img_enc and img_dec weights
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('img_enc.') or key.startswith('img_dec.'):
            filtered_state_dict[key] = value
    
    model.load_state_dict(filtered_state_dict, strict=False)
    print("Frankenstein checkpoint loaded (img_enc and img_dec weights)!")
else:
    print(f"ERROR: Frankenstein checkpoint not found at {FRANKENSTEIN_CHECKPOINT}")
    print("Please run frankensteinify.py first to create the checkpoint.")
    exit(1)

model.train()

# Optimizer - uses fp32 for optimizer state even though model is bf16
# This provides better numerical stability for momentum/variance tracking
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Loss function
criterion = nn.MSELoss()

print(f"Finetuning image autoencoder for control framework (bf16) for {NUM_STEPS} steps...")
print(f"Using random context (shape: batch x {CONTEXT_SEQ_LEN} x {EMBED_DIM}) for decoder")
print(f"Training on same images as control framework (G.random_full_image_set)")
print(f"Checkpoints saved every {SAVE_EVERY} steps")
print(f"Losses logged to {LEDGER_PATH}")
print(f"Gradient clip value: {GRADIENT_CLIP_VALUE}")

for step in range(NUM_STEPS):
    # Generate game images (bf16) - same as control framework
    img_batch = get_control_images(BATCH_SIZE, device=device, dtype=DEFAULT_DTYPE)
    
    # Forward pass (decoder gets random context)
    reconstructed = model(img_batch)
    
    # Compute loss
    loss = criterion(reconstructed, img_batch)
    
    # Check for nan/inf
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"WARNING: Loss is {loss.item()} at step {step+1}, skipping...")
        optimizer.zero_grad()
        continue
    
    # Backward pass (no gradient scaling needed for bf16)
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
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"finetune_control_step_{step+1:06d}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        save_demo_image(model, step + 1, device)
        print(f"Demo image saved for step {step+1}")

print("Training complete!")

# Final eval
model.eval()
with torch.no_grad():
    test_imgs = get_control_images(4, device=device, dtype=DEFAULT_DTYPE)
    
    test_recon = model(test_imgs)
    test_loss = criterion(test_recon, test_imgs)
    
    print(f"Final eval MSE: {test_loss.item():.6f}")

# Save final checkpoint
final_path = os.path.join(CHECKPOINT_DIR, "finetune_control_final.pt")
torch.save(model.state_dict(), final_path)
print(f"Final finetuned model saved to {final_path}")
