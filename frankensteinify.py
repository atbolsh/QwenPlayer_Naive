"""Frankensteinify: Combine trained image autoencoder with fresh QwenAgentPlayer

This script:
1. Loads the trained naked image autoencoder weights from a checkpoint
2. Creates a new QwenAgentPlayer (loading Qwen layers from HuggingFace with tied weights)
3. Attaches the trained vision encoder and decoder to the QwenAgentPlayer
4. Saves the combined model (includes layer_scale_factors for per-layer image scaling)

Note: The saved model uses tied weights (embed_tokens and lm_head share weights).
"""

import os
import torch

from visual_transformer import QwenAgentPlayer
from visual_transformer.model import ImageTransformerEncoder, ImageTransformerDecoder

# ============================================================
# EASILY EDITABLE: Input and output checkpoint paths
# ============================================================
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "brain_checkpoints")

# Input: trained image autoencoder checkpoint (img_enc and img_dec weights)
NAKED_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "finetune_control_better_embeddings_bf16_step_000200.pt")

# Output: combined frankenstein model
OUTPUT_PATH = os.path.join(CHECKPOINT_DIR, "frankenstein_with_scales_bf16.pt")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Step 1: Load the trained naked image autoencoder weights
print(f"Loading naked image autoencoder from {NAKED_CHECKPOINT}...")
naked_state_dict = torch.load(NAKED_CHECKPOINT, map_location=device)
print(f"Loaded checkpoint with keys: {list(naked_state_dict.keys())[:10]}...")

# Step 2: Create a brand new QwenAgentPlayer (loads Qwen from HuggingFace)
print("Creating fresh QwenAgentPlayer (loading Qwen3-0.6B from HuggingFace)...")
model = QwenAgentPlayer(
    model_name="Qwen/Qwen3-0.6B",
    embed_dim=1024,
    num_heads=8,
    device=device,
)
print("QwenAgentPlayer created!")

# Step 3: Attach the trained vision encoder and decoder
# The naked checkpoint has keys like 'img_enc.xxx' and 'img_dec.xxx'
# The QwenAgentPlayer has: model.pipe.model.img_enc and model.pipe.model.img_dec

print("Attaching trained vision encoder and decoder...")

# Extract img_enc weights from naked checkpoint
img_enc_state_dict = {
    k.replace('img_enc.', ''): v 
    for k, v in naked_state_dict.items() 
    if k.startswith('img_enc.')
}
print(f"  img_enc keys: {len(img_enc_state_dict)} parameters")

# Extract img_dec weights from naked checkpoint
img_dec_state_dict = {
    k.replace('img_dec.', ''): v 
    for k, v in naked_state_dict.items() 
    if k.startswith('img_dec.')
}
print(f"  img_dec keys: {len(img_dec_state_dict)} parameters")

# Load into the QwenAgentPlayer's vision components
# The structure is: model.pipe.model.img_enc and model.pipe.model.img_dec
model.pipe.model.img_enc.load_state_dict(img_enc_state_dict)
model.pipe.model.img_dec.load_state_dict(img_dec_state_dict)
print("Vision encoder and decoder weights loaded successfully!")

# Step 4: Save the combined model
# Save the entire QwenExtension (which includes Qwen model + vision components)
print(f"Saving frankenstein model to {OUTPUT_PATH}...")
torch.save(model.pipe.model.state_dict(), OUTPUT_PATH)
print("Done!")

# Verify the save
print("\nVerifying saved checkpoint...")
loaded_state_dict = torch.load(OUTPUT_PATH, map_location='cpu', weights_only=True)
print(f"Saved checkpoint has {len(loaded_state_dict)} keys")
print(f"Sample keys: {list(loaded_state_dict.keys())[:5]}...")
print(f"\nFrankenstein model saved to: {OUTPUT_PATH}")
