"""Train: Image autoencoder using img_autoencoder + L2 (MSE) Loss"""

import os
import torch
import torch.nn as nn
from general_framework import model, device, get_images, get_settings_batch

# Checkpoint directory
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "brain_checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_STEPS = 100
PRINT_EVERY = 10

# Model comes from general_framework.py (already initialized)
model.train()

# Optimizer and loss (L2 = MSE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

print(f"Training image autoencoder for {NUM_STEPS} steps...")

for step in range(NUM_STEPS):
    # Generate game images
    settings = get_settings_batch(BATCH_SIZE)
    img_batch = get_images(settings, device=device)
    
    # Forward: img_autoencoder returns reconstructed image
    reconstructed = model.img_autoencoder(img_batch, context=None)
    
    # L2 loss
    loss = criterion(reconstructed, img_batch)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (step + 1) % PRINT_EVERY == 0:
        print(f"Step {step+1}/{NUM_STEPS} | MSE Loss: {loss.item():.4f}")

print("Training complete!")

# Quick eval
model.eval()
with torch.no_grad():
    test_settings = get_settings_batch(4)
    test_imgs = get_images(test_settings, device=device)
    test_recon = model.img_autoencoder(test_imgs)
    test_loss = criterion(test_recon, test_imgs)
    print(f"Final eval MSE: {test_loss.item():.4f}")

# Save checkpoint
checkpoint_path = os.path.join(CHECKPOINT_DIR, "image_autoencoder_checkpoint.pt")
torch.save(model.state_dict(), checkpoint_path)
print(f"Model checkpoint saved to {checkpoint_path}")
