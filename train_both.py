"""Train: Text + Image jointly using forward function (LoRA version)

Uses LoRA on text components with batch_size=1 for memory efficiency.
Calls model.reset() every 3 steps to clear memory/canvases.
"""

import os
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from general_framework import (
    model, device, sdt, tokenizer,
    get_settings_batch, get_images, img_criterion
)

# Checkpoint directory
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "brain_checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 1  # Keep small for memory
LEARNING_RATE = 1e-4
NUM_STEPS = 100
PRINT_EVERY = 10
RESET_EVERY = 3  # Reset memory/canvases every N steps

# Loss weighting
TEXT_LOSS_WEIGHT = 1.0
IMG_LOSS_WEIGHT = 1.0

# LoRA config (no task_type since QwenBastardBrain is a custom model)
lora_config = LoraConfig(
    r=4, lora_alpha=16, lora_dropout=0.1, bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)

# Wrap model with LoRA
print("Applying LoRA to model...")
peft_model = get_peft_model(model, lora_config)

# Unfreeze non-Qwen components for full backward passes
# LoRA only applies to text_enc and text_dec (Qwen-inherited layers)
# All other components should train normally
# non_lora_modules = [
#     "img_enc", "img_dec", "img_weight", 
#     "dopamine", "context_tagging", "mem_enc"
# ]
# for name, param in peft_model.named_parameters():
#     for module_name in non_lora_modules:
#         if module_name in name:
#             param.requires_grad = True
#             break

peft_model.print_trainable_parameters()

peft_model.train()

# Optimizer and loss
optimizer = torch.optim.AdamW(peft_model.parameters(), lr=LEARNING_RATE)
text_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id or 0)

print(f"Training text + image jointly (LoRA) for {NUM_STEPS} steps...")

for step in range(NUM_STEPS):
    # Reset memory and canvases every RESET_EVERY steps
    if step % RESET_EVERY == 0:
        peft_model.reset()
    
    # Sample text batch
    indices = torch.randint(0, len(sdt), (BATCH_SIZE,))
    text_batch = torch.stack([sdt[i] for i in indices]).to(device)
    
    # Generate random game images
    settings_batch = get_settings_batch(BATCH_SIZE)
    img_batch = get_images(settings_batch, device=device)
    
    # Forward: get both text logits and image reconstruction
    text_logits, img_recon = peft_model.forward(
        text_batch, 
        img_batch=img_batch, 
        ret_imgs=True, 
        return_full=True, 
        use_masks=True,
        create_context=True
    )
    
    # Text loss: predict next token from current (shift by 1)
    # logits are [batch, vocab, seq_len], targets are [batch, seq_len]
    text_loss = text_criterion(text_logits[:, :, :-1], text_batch[:, 1:])
    
    # Image loss: reconstruction error
    img_loss = img_criterion(img_recon, img_batch)
    
    # Combined loss
    total_loss = TEXT_LOSS_WEIGHT * text_loss + IMG_LOSS_WEIGHT * img_loss
    
    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    model.soft_reset()

    if (step + 1) % PRINT_EVERY == 0:
        print(f"Step {step+1}/{NUM_STEPS} | Total: {total_loss.item():.4f} | Text: {text_loss.item():.4f} | Img: {img_loss.item():.4f}")

print("Training complete!")

# Quick eval
peft_model.eval()
peft_model.reset()
with torch.no_grad():
    test_text = torch.stack([sdt[i] for i in range(min(4, BATCH_SIZE))]).to(device)
    test_settings = get_settings_batch(test_text.size(0))
    test_img = get_images(test_settings, device=device)
    
    test_logits, test_recon = peft_model.forward(
        test_text, 
        img_batch=test_img, 
        ret_imgs=True, 
        return_full=True
    )
    
    test_text_loss = text_criterion(test_logits[:, :, :-1], test_text[:, 1:])
    test_img_loss = img_criterion(test_recon, test_img)
    print(f"Final eval - Text loss: {test_text_loss.item():.4f} | Img loss: {test_img_loss.item():.4f}")

# Save LoRA adapter only
lora_checkpoint_path = os.path.join(CHECKPOINT_DIR, "both_lora_adapter")
peft_model.save_pretrained(lora_checkpoint_path)
print(f"LoRA adapter saved to {lora_checkpoint_path}")

# Merge LoRA into base model and save full QwenBastardBrain
print("Merging LoRA weights into base model...")
merged_model = peft_model.merge_and_unload()
merged_checkpoint_path = os.path.join(CHECKPOINT_DIR, "both_lora_merged.pt")
torch.save(merged_model.state_dict(), merged_checkpoint_path)
print(f"Merged model checkpoint saved to {merged_checkpoint_path}")

