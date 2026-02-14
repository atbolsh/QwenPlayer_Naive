#!/usr/bin/env python3
"""
Initialize the VisionWeightedSum network in QwenAgentPlayer.

Loads the v6 checkpoint, freezes all parameters except img_weight,
trains img_weight on the control task until img_loss < 0.0006 (or 1000 batches),
then saves the full model state dict.
"""

import torch
import torch.optim as optim

# Import from frameworks (loads model, tokenizer, datasets, etc.)
from frameworks import device, create_model, tokenizer
from frameworks.control import control_batch

# ---- Config ----
V6_CHECKPOINT = "brain_checkpoints/qwen_agent_blue_line_v6_merged_batch133000.pth"
SAVE_PATH = "brain_checkpoints/qwen_agent_vision_weights_initialized.pth"
MAX_BATCHES = 1000
IMG_LOSS_TARGET = 0.0006
LR = 1e-4
BATCH_SIZE = 90

# ---- Load model ----
print(f"Creating model...")
model = create_model(device=device, use_lora=False)

print(f"Loading v6 checkpoint: {V6_CHECKPOINT}")
ckpt = torch.load(V6_CHECKPOINT, weights_only=True, map_location=device)
# Convert bf16 -> float32 for backwards compatibility
for k in list(ckpt.keys()):
    if ckpt[k].dtype == torch.bfloat16:
        ckpt[k] = ckpt[k].float()

load_result = model.pipe.model.load_state_dict(ckpt, strict=False)
if load_result.missing_keys:
    print(f"  Missing keys (fresh init): {load_result.missing_keys}")
if load_result.unexpected_keys:
    print(f"  Unexpected keys (ignored): {load_result.unexpected_keys}")

# ---- Freeze everything, unfreeze only img_weight ----
for param in model.pipe.model.parameters():
    param.requires_grad = False

for param in model.pipe.model.img_weight.parameters():
    param.requires_grad = True

# Verify
n_trainable = sum(p.numel() for p in model.pipe.model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.pipe.model.parameters())
print(f"Trainable parameters: {n_trainable:,} / {n_total:,} total")

# ---- Optimizer (only img_weight params) ----
optimizer = optim.Adam(model.pipe.model.img_weight.parameters(), lr=LR, eps=1e-9)

# ---- Training loop ----
print(f"\nTraining img_weight for up to {MAX_BATCHES} batches (target img_loss < {IMG_LOSS_TARGET})...")
print(f"  batch_size={BATCH_SIZE}, lr={LR}\n")

for b in range(MAX_BATCHES):
    # Reset every 3rd batch (same cadence as general_training.py)
    # so canvases can accumulate between resets
    reset_model = (b % 3 == 2)

    print(f"Batch {b+1}:")
    loss, text_loss, img_loss = control_batch(
        batch_size=BATCH_SIZE,
        model=model,
        optimizer=optimizer,
        batch_num=b,
        compute_grad=True,
        training=True,
        model_eval=False,
        reset_model=reset_model,
        printing=True,
    )

    if img_loss < IMG_LOSS_TARGET and len(model.canvases) >= 2 and b > 50:
        print(f"\n=== Target reached at batch {b + 1}: img_loss={img_loss:.6f} < {IMG_LOSS_TARGET}, canvases={len(model.canvases)} ===")
        break

# ---- Unfreeze all and save full state dict ----
for param in model.pipe.model.parameters():
    param.requires_grad = True

state_dict = model.pipe.model.state_dict()
torch.save(state_dict, SAVE_PATH)
print(f"\nSaved full model state dict to {SAVE_PATH}")
print(f"  Keys: {len(state_dict)}")
print("Done!")
