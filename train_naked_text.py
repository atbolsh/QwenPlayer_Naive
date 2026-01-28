"""Train: Naked text model (Qwen3 only) with periodic checkpoints

Uses general_framework_lightweight for text utilities without loading the vision components.
This trains just the Qwen3 language model on next-token prediction.
"""

import os
import csv
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM

from general_framework_lightweight import (
    device, tokenizer, load_text_datasets, QWEN_MODEL_NAME
)

# Directories
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "brain_checkpoints")
LEDGER_PATH = os.path.join(os.path.dirname(__file__), "naked_text_losses.csv")
# Set to None to start fresh, or provide path to resume
LOAD_CHECKPOINT = None
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hyperparameters, optimized for H100
BATCH_SIZE = 750
LEARNING_RATE = 1e-5
NUM_STEPS = 10000
PRINT_EVERY = 100
SAVE_EVERY = 1000

# Initialize CSV ledger
with open(LEDGER_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'loss'])

# Load datasets
print("Loading text datasets...")
sdt, sdv = load_text_datasets()
print(f"Loaded {len(sdt)} training examples")

# Initialize model (just Qwen3, no vision)
print(f"Loading {QWEN_MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map=device,
)

# Resize embeddings if we added special tokens
model.resize_token_embeddings(len(tokenizer))

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
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id or 0)

print(f"Training naked text model for {NUM_STEPS} steps...")
print(f"Checkpoints saved every {SAVE_EVERY} steps")
print(f"Losses logged to {LEDGER_PATH}")

for step in range(NUM_STEPS):
    # Sample batch
    indices = torch.randint(0, len(sdt), (BATCH_SIZE,))
    text_batch = torch.stack([sdt[i] for i in indices]).to(device)
    
    # Create attention mask
    pad_token_id = tokenizer.pad_token_id or 0
    attention_mask = (text_batch != pad_token_id).long()
    
    # Forward
    outputs = model(
        input_ids=text_batch,
        attention_mask=attention_mask,
    )
    
    # logits: (batch, seq_len, vocab)
    logits = outputs.logits
    
    # Loss: predict next token from current (shift by 1)
    # logits[:, :-1, :] predicts tokens 1..N from input tokens 0..N-1
    loss = criterion(
        logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
        text_batch[:, 1:].contiguous().view(-1)
    )
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if ((step + 1) % PRINT_EVERY == 0) or (step < 10):
        loss_val = loss.item()
        print(f"Step {step+1}/{NUM_STEPS} | Loss: {loss_val:.6f}")
        
        # Log to CSV
        with open(LEDGER_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step + 1, loss_val])
    
    # Save checkpoint every SAVE_EVERY steps
    if (step + 1) % SAVE_EVERY == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"naked_text_step_{step+1:06d}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

print("Training complete!")

# Final eval
model.eval()
with torch.no_grad():
    test_batch = torch.stack([sdt[i] for i in range(min(4, len(sdt)))]).to(device)
    pad_token_id = tokenizer.pad_token_id or 0
    attention_mask = (test_batch != pad_token_id).long()
    
    outputs = model(input_ids=test_batch, attention_mask=attention_mask)
    logits = outputs.logits
    
    test_loss = criterion(
        logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
        test_batch[:, 1:].contiguous().view(-1)
    )
    print(f"Final eval loss: {test_loss.item():.6f}")

# Save final checkpoint
final_path = os.path.join(CHECKPOINT_DIR, "naked_text_final.pt")
torch.save(model.state_dict(), final_path)
print(f"Final model saved to {final_path}")
