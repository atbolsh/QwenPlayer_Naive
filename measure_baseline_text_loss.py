"""Measure baseline cross-entropy loss of untrained Qwen3-0.6B on control text samples.

This provides a reference point to compare against trained QwenAgent systems.
If the trained system has worse loss than this baseline, something is broken
or significant fine-tuning is needed.

Results are printed and saved to baseline_text_loss.txt
"""

import os
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
load_dotenv()

from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the control framework's text dataset
from general_framework_lightweight import (
    load_text_datasets, get_text_batch, tokenizer, device, MAX_SEQ_LENGTH
)

# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "Qwen/Qwen3-0.6B"
BATCH_SIZE = 32
NUM_BATCHES = 10  # Evaluate on multiple batches for stability
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "baseline_text_loss.txt")

print(f"Device: {device}")
print(f"Model: {MODEL_NAME}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Num batches: {NUM_BATCHES}")
print()

# ============================================================
# Load untrained Qwen3 model
# ============================================================
print("Loading fresh Qwen3-0.6B model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map=device,
)
model.eval()
print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# Load text dataset
# ============================================================
print("\nLoading text dataset...")
sdt, sdv = load_text_datasets()
num_samples = len(sdt)
print(f"Dataset loaded with {num_samples} samples")

# ============================================================
# Compute cross-entropy loss
# ============================================================
def compute_text_loss(model, input_ids):
    """
    Compute cross-entropy loss for next-token prediction.
    
    Args:
        model: The language model
        input_ids: Token IDs (batch_size, seq_len)
    
    Returns:
        Average cross-entropy loss (scalar)
    """
    with torch.no_grad():
        # Forward pass
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)
        
        # Shift for next-token prediction
        # Predict token[i+1] from position i
        shift_logits = logits[:, :-1, :].contiguous()  # (batch, seq-1, vocab)
        shift_labels = input_ids[:, 1:].contiguous()    # (batch, seq-1)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean'
        )
        
        return loss.item()


print("\nMeasuring baseline loss...")
print("-" * 50)

losses = []
for batch_idx in range(NUM_BATCHES):
    # Get batch of text samples
    start_idx = (batch_idx * BATCH_SIZE) % num_samples
    if start_idx + BATCH_SIZE > num_samples:
        start_idx = num_samples - BATCH_SIZE
    
    text_batch = get_text_batch(sdt, start_idx, BATCH_SIZE, target_device=device)
    
    # Compute loss
    loss = compute_text_loss(model, text_batch)
    losses.append(loss)
    
    print(f"Batch {batch_idx + 1}/{NUM_BATCHES}: loss = {loss:.4f}")

# ============================================================
# Compute statistics
# ============================================================
import statistics

mean_loss = statistics.mean(losses)
std_loss = statistics.stdev(losses) if len(losses) > 1 else 0.0
min_loss = min(losses)
max_loss = max(losses)

print()
print("=" * 50)
print("BASELINE RESULTS (Untrained Qwen3-0.6B)")
print("=" * 50)
print(f"Mean loss:   {mean_loss:.4f}")
print(f"Std dev:     {std_loss:.4f}")
print(f"Min loss:    {min_loss:.4f}")
print(f"Max loss:    {max_loss:.4f}")
print(f"Perplexity:  {torch.exp(torch.tensor(mean_loss)).item():.2f}")
print("=" * 50)

# ============================================================
# Save results
# ============================================================
with open(OUTPUT_FILE, 'w') as f:
    f.write("Baseline Text Cross-Entropy Loss (Untrained Qwen3-0.6B)\n")
    f.write("=" * 50 + "\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Dataset: control text samples (sdt/ProcessBench)\n")
    f.write(f"Batch size: {BATCH_SIZE}\n")
    f.write(f"Num batches: {NUM_BATCHES}\n")
    f.write(f"Total samples: {BATCH_SIZE * NUM_BATCHES}\n")
    f.write("\n")
    f.write("Results:\n")
    f.write(f"  Mean loss:   {mean_loss:.4f}\n")
    f.write(f"  Std dev:     {std_loss:.4f}\n")
    f.write(f"  Min loss:    {min_loss:.4f}\n")
    f.write(f"  Max loss:    {max_loss:.4f}\n")
    f.write(f"  Perplexity:  {torch.exp(torch.tensor(mean_loss)).item():.2f}\n")
    f.write("\n")
    f.write("Individual batch losses:\n")
    for i, loss in enumerate(losses):
        f.write(f"  Batch {i+1}: {loss:.4f}\n")
    f.write("\n")
    f.write("Interpretation:\n")
    f.write("  - If trained QwenAgent text loss >> this baseline, something is broken\n")
    f.write("  - If trained QwenAgent text loss ~= this baseline, model preserved capabilities\n")
    f.write("  - If trained QwenAgent text loss << this baseline, model improved on this task\n")

print(f"\nResults saved to: {OUTPUT_FILE}")
