"""Train: Text autoencoder using sentence_autoencoder + CrossEntropyLoss (LoRA version)"""

import os
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType

from general_framework import model, device, sdt, tokenizer

# Checkpoint directory
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "brain_checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_STEPS = 100
PRINT_EVERY = 10

# LoRA config
lora_config = LoraConfig(
    r=4, lora_alpha=16, lora_dropout=0.1, bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type=TaskType.CAUSAL_LM
)

# Wrap model with LoRA
print("Applying LoRA to model...")
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

peft_model.train()

# Optimizer and loss (only LoRA params are trainable)
optimizer = torch.optim.AdamW(peft_model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id or 0)

print(f"Training text autoencoder (LoRA) for {NUM_STEPS} steps...")

for step in range(NUM_STEPS):
    # Sample batch
    indices = torch.randint(0, len(sdt), (BATCH_SIZE,))
    text_batch = torch.stack([sdt[i] for i in indices]).to(device)
    
    # Forward: sentence_autoencoder returns [batch, vocab, seq_len]
    logits = peft_model.sentence_autoencoder(text_batch, context=None, return_full=True, use_masks=True)
    
    # Loss: predict next token from current (shift by 1)
    # logits[:, :, :-1] predicts tokens 1..N from input tokens 0..N-1
    loss = criterion(logits[:, :, :-1], text_batch[:, 1:])
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (step + 1) % PRINT_EVERY == 0:
        print(f"Step {step+1}/{NUM_STEPS} | Loss: {loss.item():.4f}")

print("Training complete!")

# Quick eval
peft_model.eval()
with torch.no_grad():
    test_batch = torch.stack([sdt[i] for i in range(4)]).to(device)
    logits = peft_model.sentence_autoencoder(test_batch, return_full=True, use_masks=True)
    test_loss = criterion(logits[:, :, :-1], test_batch[:, 1:])
    print(f"Final eval loss: {test_loss.item():.4f}")

# Save LoRA adapter only
lora_checkpoint_path = os.path.join(CHECKPOINT_DIR, "text_autoencoder_lora_adapter")
peft_model.save_pretrained(lora_checkpoint_path)
print(f"LoRA adapter saved to {lora_checkpoint_path}")

# Merge LoRA into base model and save full QwenBastardBrain
print("Merging LoRA weights into base model...")
merged_model = peft_model.merge_and_unload()
merged_checkpoint_path = os.path.join(CHECKPOINT_DIR, "text_autoencoder_lora_merged.pt")
torch.save(merged_model.state_dict(), merged_checkpoint_path)
print(f"Merged model checkpoint saved to {merged_checkpoint_path}")

