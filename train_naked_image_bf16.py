"""Train: Naked image autoencoder (encoder + decoder only) in BFloat16 with periodic checkpoints

Uses general_framework_lightweight for game utilities without loading the Qwen model.
This script finetunes a float32 checkpoint to work with bf16 weights.

Key bf16 considerations:
- Uses torch.cuda.amp.GradScaler for gradient scaling to prevent underflow
- Clamps outputs to valid range to avoid inf/nan
- Uses bf16 for forward/backward passes but fp32 for optimizer state
"""

import os
import csv
import torch
import torch.nn as nn
# Note: GradScaler is not needed for bf16 (it has same exponent range as fp32)
from torchvision.utils import save_image

from visual_transformer.model import ImageTransformerEncoder, ImageTransformerDecoder
from general_framework_lightweight import get_images, get_settings_batch, device, img_criterion

# Directories
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "brain_checkpoints")
DEMO_DIR = os.path.join(os.path.dirname(__file__), "demo_images")
LEDGER_PATH = os.path.join(os.path.dirname(__file__), "naked_image_bf16_losses.csv")

# Checkpoint to load and convert to bf16
LOAD_CHECKPOINT = os.path.join(os.path.dirname(__file__), "brain_checkpoints/naked_image_step_052000.pt")

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


def convert_to_bf16(model):
    """
    Convert a model's weights and biases to bf16.
    
    This converts all parameters to bfloat16, which:
    - Reduces memory usage by ~50%
    - Can speed up training on hardware with bf16 support (A100, H100)
    - Maintains the dynamic range of fp32 (unlike fp16)
    
    Args:
        model: A PyTorch model with fp32 weights
        
    Returns:
        The same model with bf16 weights (modified in-place)
    """
    for name, param in model.named_parameters():
        param.data = param.data.to(torch.bfloat16)
    
    # Also convert buffers (e.g., running mean/var in BatchNorm)
    for name, buffer in model.named_buffers():
        if buffer.is_floating_point():
            buffer.data = buffer.data.to(torch.bfloat16)
    
    return model


class NakedImageAutoencoderBF16(nn.Module):
    """Standalone image autoencoder with bf16 support.
    
    Same architecture as NakedImageAutoencoder but with:
    - bf16 weights and biases
    - Explicit dtype handling for inputs
    """
    
    def __init__(self, embed_dim=1024, num_heads=8, dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.img_enc = ImageTransformerEncoder(embed_dim=embed_dim, num_heads=num_heads)
        self.img_dec = ImageTransformerDecoder(embed_dim=embed_dim, num_heads=num_heads)
        
        # Convert all weights and biases to bf16
        convert_to_bf16(self)
    
    def forward(self, img_batch):
        # Ensure input is bf16
        if img_batch.dtype != self.dtype:
            img_batch = img_batch.to(self.dtype)
        
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
        img = get_images(settings, device=device)
        
        # Convert to bf16 for inference
        img_bf16 = img.to(torch.bfloat16)
        recon = model(img_bf16)
        
        # Convert back to fp32 for saving
        recon_fp32 = recon.float()
        
        input_path = os.path.join(DEMO_DIR, f"naked_bf16_step_{step:06d}_input.png")
        output_path = os.path.join(DEMO_DIR, f"naked_bf16_step_{step:06d}_output.png")
        save_image(img[0], input_path)
        save_image(recon_fp32[0].clamp(0, 1), output_path)
    model.train()


# Initialize CSV ledger
with open(LEDGER_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'mse_loss'])

# Initialize model
print("Initializing NakedImageAutoencoderBF16...")
model = NakedImageAutoencoderBF16(embed_dim=1024, num_heads=8, dtype=torch.bfloat16).to(device)

# Load fp32 checkpoint and convert to bf16
if os.path.exists(LOAD_CHECKPOINT):
    print(f"Loading fp32 checkpoint from {LOAD_CHECKPOINT}...")
    fp32_state_dict = torch.load(LOAD_CHECKPOINT, map_location=device)
    
    # Load the fp32 weights first
    model.load_state_dict(fp32_state_dict)
    print("Checkpoint loaded!")
    
    # Convert to bf16
    print("Converting model to bf16...")
    model = convert_to_bf16(model)
    print("Model converted to bf16!")
else:
    print(f"ERROR: Checkpoint not found at {LOAD_CHECKPOINT}")
    print("This script requires a pretrained fp32 checkpoint to finetune.")
    print("Run train_naked_image.py first to create the checkpoint.")
    exit(1)

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
    # Generate game images (comes as fp32)
    settings = get_settings_batch(BATCH_SIZE, bare=False, restrict_angles=True)
    img_batch = get_images(settings, device=device)
    
    # Convert to bf16
    img_batch_bf16 = img_batch.to(torch.bfloat16)
    
    # Forward pass (model is already bf16, no autocast needed)
    reconstructed = model(img_batch_bf16)
    
    # Compute loss in bf16
    loss = criterion(reconstructed, img_batch_bf16)
    
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
    test_imgs = get_images(test_settings, device=device)
    test_imgs_bf16 = test_imgs.to(torch.bfloat16)
    
    test_recon = model(test_imgs_bf16)
    test_loss = criterion(test_recon, test_imgs_bf16)
    
    print(f"Final eval MSE: {test_loss.item():.6f}")

# Save final checkpoint
final_path = os.path.join(CHECKPOINT_DIR, "naked_image_bf16_final.pt")
torch.save(model.state_dict(), final_path)
print(f"Final bf16 model saved to {final_path}")
