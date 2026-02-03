"""Check activation scales at different stages of the pipeline."""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
load_dotenv()

from visual_transformer.model import ImageTransformerEncoder
from general_framework_lightweight import G, device

print("="*60)
print("ACTIVATION SCALE ANALYSIS")
print("="*60)

# ============================================================
# 1. Qwen3 embedding layer output (input to transformers)
# ============================================================
print("\n[1] Loading Qwen3-0.6B...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
qwen_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    torch_dtype=torch.bfloat16,
    tie_word_embeddings=False
).to(device)

# Get some sample text
sample_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models process text tokens.",
    "What is the meaning of life?",
]

print("\n--- Embedding Layer Output (Input to Transformers) ---")
for text in sample_texts:
    tokens = tokenizer(text, return_tensors='pt')['input_ids'].to(device)
    
    with torch.no_grad():
        # Get raw embeddings from embed_tokens
        embeddings = qwen_model.model.embed_tokens(tokens)  # (1, seq_len, 1024)
    
    emb_np = embeddings[0].float().cpu().numpy()
    per_token_mags = np.linalg.norm(emb_np, axis=1)
    
    print(f"Text: '{text[:40]}...'")
    print(f"  Per-token magnitude: mean={per_token_mags.mean():.4f}, max={per_token_mags.max():.4f}")
    print(f"  Max component value: {emb_np.max():.4f}")
    print(f"  Std of components: {emb_np.std():.4f}")

# ============================================================
# 2. Qwen3 forward output (input to lm_head, post-norm)
# ============================================================
print("\n--- Forward Output (Input to lm_head, post-RMSNorm) ---")
for text in sample_texts:
    tokens = tokenizer(text, return_tensors='pt')['input_ids'].to(device)
    
    with torch.no_grad():
        # Get hidden states from base model (includes final RMSNorm)
        outputs = qwen_model.model(tokens, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state  # (1, seq_len, 1024)
    
    hs_np = hidden_states[0].float().cpu().numpy()
    per_token_mags = np.linalg.norm(hs_np, axis=1)
    
    print(f"Text: '{text[:40]}...'")
    print(f"  Per-token magnitude: mean={per_token_mags.mean():.4f}, max={per_token_mags.max():.4f}")
    print(f"  Max component value: {hs_np.max():.4f}")
    print(f"  Std of components: {hs_np.std():.4f}")

# Unload Qwen3
del qwen_model
import gc
gc.collect()
torch.cuda.empty_cache()
print("\nQwen3 unloaded.")

# ============================================================
# 3. Image encoder output
# ============================================================
print("\n[3] Loading image encoder...")

# Load image encoder (float32 for this checkpoint)
img_enc = ImageTransformerEncoder(embed_dim=1024, num_heads=8, dtype=torch.float32).to(device)

# Load weights from available checkpoint
import os
checkpoint_path = "brain_checkpoints/naked_image_step_052000.pt"

print(f"Loading from: {checkpoint_path}")
state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
enc_dict = {k.replace('img_enc.', ''): v for k, v in state_dict.items() if k.startswith('img_enc.')}
img_enc.load_state_dict(enc_dict)
img_enc.eval()

print("\n--- Image Encoder Output ---")
# Generate some game images
for i in range(3):
    img_set = G.random_full_image_set(restrict_angles=True)
    # Take first image
    img = img_set[0:1]  # (1, 224, 224, 3)
    img_tensor = torch.permute(torch.tensor(img, dtype=torch.float32).to(device), (0, 3, 1, 2))
    
    with torch.no_grad():
        encoding = img_enc(img_tensor)  # (1, 256, 1024)
    
    enc_np = encoding[0].float().cpu().numpy()
    per_token_mags = np.linalg.norm(enc_np, axis=1)
    
    print(f"Image {i+1}:")
    print(f"  Per-token magnitude: mean={per_token_mags.mean():.4f}, max={per_token_mags.max():.4f}")
    print(f"  Max component value: {enc_np.max():.4f}")
    print(f"  Std of components: {enc_np.std():.4f}")

# ============================================================
# Summary comparison
# ============================================================
print("\n" + "="*60)
print("SUMMARY COMPARISON")
print("="*60)
print("""
                              Per-Token Magnitude    Max Component
Qwen3 embed_tokens output:         ~0.8               ~0.08
Qwen3 forward output (post-norm):  ~80-100            ~25-35
Image encoder output:              ~32                ~5

Key insight: There's a ~100x scale difference between:
  - Embedding layer output (~0.8)
  - Post-transformer output (~80-100)
  
The RMSNorm doesn't "normalize" to magnitude 1; it normalizes 
the root-mean-square to 1, which with 1024 dimensions gives
magnitude ~sqrt(1024) = 32... but the actual output is even larger!
""")
