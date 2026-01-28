"""Train: Naked image autoencoder (encoder + decoder only) in BFloat16 with periodic checkpoints

Uses general_framework_lightweight for game utilities without loading the Qwen model.
This script can finetune an existing checkpoint or start fresh.

Note: ImageTransformerEncoder/Decoder now default to bf16, so no explicit conversion needed.
"""

import os
import csv
import torch
import torch.nn as nn
from torchvision.utils import save_image

from visual_transformer.model import ImageTransformerEncoder, ImageTransformerDecoder
from general_framework_lightweight import get_images, get_settings_batch, device, img_criterion

# Default dtype
DEFAULT_DTYPE = torch.bfloat16

# Directories
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "brain_checkpoints")
DEMO_DIR = os.path.join(os.path.dirname(__file__), "demo_images")
LEDGER_PATH = os.path.join(os.path.dirname(__file__), "naked_image_bf16_losses.csv")

# Checkpoint to load (optional - set to None to start fresh)
LOAD_CHECKPOINT = None  # os.path.join(os.path.dirname(__file__), "brain_checkpoints/naked_image_bf16_step_008000.pt")

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


class NakedImageAutoencoderBF16(nn.Module):
    """Standalone image autoencoder in bf16.
    
    ImageTransformerEncoder/Decoder now default to bf16, so this is just a wrapper.
    """
    
    def __init__(self, embed_dim=1024, num_heads=8, dtype=DEFAULT_DTYPE):
        super().__init__()
        self.dtype = dtype
        # img_enc and img_dec now default to bf16
        self.img_enc = ImageTransformerEncoder(embed_dim=embed_dim, num_heads=num_heads, dtype=dtype)
        self.img_dec = ImageTransformerDecoder(embed_dim=embed_dim, num_heads=num_heads, dtype=dtype)
    
    def forward(self, img_batch):
        # img_enc/img_dec handle dtype conversion internally
        encoding = self.img_enc(img_batch)
        reconstruction = self.img_dec(encoding, encoding)
        return reconstruction
    
    def get_device(self):
        return self.img_enc.get_device()


def save_demo_image(model, step, device):
    """Save a single input-output pair."""
    model.eval()
    with torch.no_grad():
        settings = get_settings_batch(1, bare=False, restrict_angles=True)
        img = get_images(settings, device=device, dtype=DEFAULT_DTYPE)
        recon = model(img)
        
        input_path = os.path.join(DEMO_DIR, f"naked_bf16_step_{step:06d}_input.png")
        output_path = os.path.join(DEMO_DIR, f"naked_bf16_step_{step:06d}_output.png")
        # Convert to float32 for saving (torchvision requires it)
        save_image(img[0].float(), input_path)
        save_image(recon[0].float().clamp(0, 1), output_path)
    model.train()


# Initialize CSV ledger
with open(LEDGER_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'mse_loss'])

# Initialize model (bf16 by default)
print("Initializing NakedImageAutoencoderBF16...")
model = NakedImageAutoencoderBF16(embed_dim=1024, num_heads=8, dtype=DEFAULT_DTYPE).to(device)

# Load checkpoint if specified
if LOAD_CHECKPOINT and os.path.exists(LOAD_CHECKPOINT):
    print(f"Loading checkpoint from {LOAD_CHECKPOINT}...")
    state_dict = torch.load(LOAD_CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("Checkpoint loaded!")
else:
    print("Starting fresh (no checkpoint loaded)")

model.train()

# Optimizer - uses fp32 for optimizer state even though model is bf16
# This provides better numerical stability for momentum/variance tracking
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Loss function
criterion = nn.MSELoss()

print(f"Training naked image autoencoder (bf16) for {NUM_STEPS} steps...")
print(f"Checkpoints saved every {SAVE_EVERY} steps")
print(f"Losses logged to {LEDGER_PATH}")
print(f"Gradient clip value: {GRADIENT_CLIP_VALUE}")

for step in range(NUM_STEPS):
    # Generate game images (bf16)
    settings = get_settings_batch(BATCH_SIZE, bare=False, restrict_angles=True)
    img_batch = get_images(settings, device=device, dtype=DEFAULT_DTYPE)
    
    # Forward pass
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
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"naked_image_bf16_step_{step+1:06d}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        save_demo_image(model, step + 1, device)
        print(f"Demo image saved for step {step+1}")

print("Training complete!")

# Final eval
model.eval()
with torch.no_grad():
    test_settings = get_settings_batch(4, bare=False, restrict_angles=True)
    test_imgs = get_images(test_settings, device=device, dtype=DEFAULT_DTYPE)
    
    test_recon = model(test_imgs)
    test_loss = criterion(test_recon, test_imgs)
    
    print(f"Final eval MSE: {test_loss.item():.6f}")

# Save final checkpoint
final_path = os.path.join(CHECKPOINT_DIR, "naked_image_bf16_final.pt")
torch.save(model.state_dict(), final_path)
print(f"Final bf16 model saved to {final_path}")
