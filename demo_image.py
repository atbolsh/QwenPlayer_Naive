"""Demo: Image processing with QwenBastardBrain's img_autoencoder"""

import torch
import matplotlib.pyplot as plt
from general_framework import QwenBastardBrain, device, G, get_images, get_settings_batch

# Load model
model = QwenBastardBrain().to(device)
model.eval()

# Generate game images
batch_size = 4
settings = get_settings_batch(batch_size)
img_batch = get_images(settings, device=device)

print(f"Image batch shape: {img_batch.shape}")  # [batch, 3, 224, 224]

# Run image autoencoder
with torch.no_grad():
    reconstructed = model.img_autoencoder(img_batch, context=None)

print(f"Reconstructed shape: {reconstructed.shape}")

# Compute reconstruction error
mse = torch.nn.functional.mse_loss(reconstructed, img_batch)
print(f"MSE reconstruction loss: {mse.item():.4f}")

# Visualize (optional - requires display)
try:
    fig, axes = plt.subplots(2, batch_size, figsize=(3*batch_size, 6))
    for i in range(batch_size):
        # Original
        orig = img_batch[i].permute(1, 2, 0).cpu().numpy().clip(0, 1)
        axes[0, i].imshow(orig)
        axes[0, i].set_title(f"Original {i}")
        axes[0, i].axis('off')
        
        # Reconstructed
        recon = reconstructed[i].permute(1, 2, 0).cpu().numpy().clip(0, 1)
        axes[1, i].imshow(recon)
        axes[1, i].set_title(f"Recon {i}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("image_demo_output.png")
    print("Saved visualization to image_demo_output.png")
except Exception as e:
    print(f"Visualization skipped: {e}")

