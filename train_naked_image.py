"""Train: Naked image autoencoder (encoder + decoder only) with periodic checkpoints

Uses frameworks.general_framework_lightweight for game utilities without loading the Qwen model.
"""

import os
import csv
import torch
import torch.nn as nn
from torchvision.utils import save_image

from visual_transformer.model import ImageTransformerEncoder, ImageTransformerDecoder
from general_framework_lightweight import get_images, get_settings_batch, device, img_criterion

# Directories
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "brain_checkpoints")
DEMO_DIR = os.path.join(os.path.dirname(__file__), "demo_images")
LEDGER_PATH = os.path.join(os.path.dirname(__file__), "naked_ledger_losses_v2.csv")
# Set to None to start fresh, or provide path to resume
LOAD_CHECKPOINT = None  # os.path.join(os.path.dirname(__file__), "brain_checkpoints/naked_image_step_052000.pt")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DEMO_DIR, exist_ok=True)

# Hyperparameters
# Currently optimized for H100 with 96Gb in VRAM
BATCH_SIZE = 550
LEARNING_RATE = 1e-5
NUM_STEPS = 10000000
PRINT_EVERY = 100
SAVE_EVERY = 1000


class NakedImageAutoencoder(nn.Module):
    """Standalone image autoencoder with same params as QwenExtension's img_enc/img_dec."""
    
    def __init__(self, embed_dim=1024, num_heads=8):
        super().__init__()
        self.img_enc = ImageTransformerEncoder(embed_dim=embed_dim, num_heads=num_heads)
        self.img_dec = ImageTransformerDecoder(embed_dim=embed_dim, num_heads=num_heads)
    
    def forward(self, img_batch):
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
        recon = model(img)
        
        input_path = os.path.join(DEMO_DIR, f"naked_step_{step:06d}_v2_input.png")
        output_path = os.path.join(DEMO_DIR, f"naked_step_{step:06d}_v2_output.png")
        save_image(img[0], input_path)
        save_image(recon[0].clamp(0, 1), output_path)
    model.train()


# Initialize CSV ledger
with open(LEDGER_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'mse_loss'])

# Initialize model
print("Initializing NakedImageAutoencoder...")
model = NakedImageAutoencoder(embed_dim=1024, num_heads=8).to(device)

# Load from checkpoint if specified
if LOAD_CHECKPOINT and os.path.exists(LOAD_CHECKPOINT):
    print(f"Loading checkpoint from {LOAD_CHECKPOINT}...")
    model.load_state_dict(torch.load(LOAD_CHECKPOINT, map_location=device))
    print("Checkpoint loaded!")
else:
    print("Starting fresh (no checkpoint loaded)")

model.train()

# Optimizer and loss
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

print(f"Training naked image autoencoder for {NUM_STEPS} steps...")
print(f"Checkpoints saved every {SAVE_EVERY} steps")
print(f"Losses logged to {LEDGER_PATH}")

for step in range(NUM_STEPS):
    # Generate game images
    settings = get_settings_batch(BATCH_SIZE, bare=False, restrict_angles=True)
    img_batch = get_images(settings, device=device)
    
    # Forward
    reconstructed = model(img_batch)
    
    # L2 loss
    loss = criterion(reconstructed, img_batch)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
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
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"naked_image_step_{step+1:06d}_v2.pt")
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
    test_recon = model(test_imgs)
    test_loss = criterion(test_recon, test_imgs)
    print(f"Final eval MSE: {test_loss.item():.6f}")

# Save final checkpoint
final_path = os.path.join(CHECKPOINT_DIR, "naked_image_final_v2.pt")
torch.save(model.state_dict(), final_path)
print(f"Final model saved to {final_path}")
