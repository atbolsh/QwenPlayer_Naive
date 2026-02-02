"""Quick script to inspect image encoder output magnitudes."""

import torch
import numpy as np
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

device = torch.device('cuda:0')

# Import image encoder and game utilities
from visual_transformer.model import ImageTransformerEncoder, ImageTransformerDecoder
from general_framework_lightweight import G

print("Loading image encoder from naked_image_step_052000.pt...")

# Create encoder (float32 since that checkpoint is float32)
img_enc = ImageTransformerEncoder(embed_dim=1024, num_heads=8, dtype=torch.float32).to(device)
img_dec = ImageTransformerDecoder(embed_dim=1024, num_heads=8, dtype=torch.float32).to(device)

# Load weights
checkpoint_path = "brain_checkpoints/naked_image_step_052000.pt"
state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

# Filter and load
enc_dict = {k.replace('img_enc.', ''): v for k, v in state_dict.items() if k.startswith('img_enc.')}
dec_dict = {k.replace('img_dec.', ''): v for k, v in state_dict.items() if k.startswith('img_dec.')}
img_enc.load_state_dict(enc_dict)
img_dec.load_state_dict(dec_dict)
print("Weights loaded!")

# Generate some game images
print("\nGenerating game images...")
img_set = G.random_full_image_set(restrict_angles=True)
print(f"Image set shape: {img_set.shape}")  # (N, H, W, C)

# Convert to tensor (N, C, H, W)
img_tensor = torch.permute(torch.tensor(img_set, dtype=torch.float32).to(device), (0, 3, 1, 2))
print(f"Tensor shape: {img_tensor.shape}")

# Encode
img_enc.eval()
with torch.no_grad():
    encodings = img_enc(img_tensor)  # (N, 256, 1024)

print(f"\n=== Image encoding stats ===")
print(f"Encoding shape: {encodings.shape}")

# Stats across all encodings
all_enc = encodings.float().cpu().numpy()
print(f"\nAcross all encodings:")
print(f"  Max value: {all_enc.max():.4f}")
print(f"  Min value: {all_enc.min():.4f}")
print(f"  Mean: {all_enc.mean():.4f}")
print(f"  Std: {all_enc.std():.4f}")

# Per-image magnitude
print(f"\n=== Magnitude per image (first 10) ===")
for i in range(min(10, encodings.shape[0])):
    enc = encodings[i].float().cpu().numpy()  # (256, 1024)
    # Magnitude of entire encoding (flatten)
    full_mag = np.linalg.norm(enc)
    # Average magnitude per token
    per_token_mags = np.linalg.norm(enc, axis=1)  # (256,)
    avg_token_mag = per_token_mags.mean()
    max_val = enc.max()
    print(f"Image {i}: full_mag={full_mag:.2f}, avg_token_mag={avg_token_mag:.4f}, max={max_val:.4f}")

# Histogram of a single encoding's values
print(f"\n=== Histogram of first image encoding ===")
first_enc = encodings[0].float().cpu().numpy().flatten()
hist, bin_edges = np.histogram(first_enc, bins=20)
for i in range(len(hist)):
    bar = '#' * (hist[i] // 100 + 1) if hist[i] > 0 else ''
    print(f"[{bin_edges[i]:7.3f}, {bin_edges[i+1]:7.3f}): {hist[i]:5d} {bar}")

# Compare with random noise at different scales
print(f"\n=== For comparison: random noise magnitudes ===")
for scale in [1.0, 1/32, 100]:
    rand = torch.randn(1, 32, 1024) * scale
    mag = torch.norm(rand).item()
    per_token = torch.norm(rand, dim=2).mean().item()
    print(f"randn * {scale}: full_mag={mag:.2f}, avg_token_mag={per_token:.4f}")
