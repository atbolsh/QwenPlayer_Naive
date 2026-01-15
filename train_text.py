"""Train: Text autoencoder using sentence_autoencoder + CrossEntropyLoss"""

import os
import torch
import torch.nn as nn
from general_framework import model, device, sdt, tokenizer

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

# Optimizer and loss
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id or 0)

print(f"Training text autoencoder for {NUM_STEPS} steps...")

for step in range(NUM_STEPS):
    # Sample batch
    indices = torch.randint(0, len(sdt), (BATCH_SIZE,))
    text_batch = torch.stack([sdt[i] for i in indices]).to(device)
    
    # Forward: sentence_autoencoder returns [batch, vocab, seq_len]
    logits = model.sentence_autoencoder(text_batch, context=None, return_full=True, use_masks=True)
    
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
model.eval()
with torch.no_grad():
    test_batch = torch.stack([sdt[i] for i in range(4)]).to(device)
    logits = model.sentence_autoencoder(test_batch, return_full=True, use_masks=True)
    test_loss = criterion(logits[:, :, :-1], test_batch[:, 1:])
    print(f"Final eval loss: {test_loss.item():.4f}")

# Save checkpoint
checkpoint_path = os.path.join(CHECKPOINT_DIR, "text_autoencoder_checkpoint.pt")
torch.save(model.state_dict(), checkpoint_path)
print(f"Model checkpoint saved to {checkpoint_path}")
