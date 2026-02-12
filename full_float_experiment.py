#!/usr/bin/env python3
"""Full Float Experiment

Compare Qwen3-0.6B performance in bf16 vs float32 on the ProcessBench dataset.
Measures cross-entropy loss on next-token prediction.
"""

import torch
import torch.nn as nn
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Configuration
MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = torch.device("cuda:0")
NUM_SAMPLES = 100  # Number of samples to evaluate
BATCH_SIZE = 4
MAX_SEQ_LENGTH = 128

print("=" * 70)
print("Full Float Experiment: bf16 vs float32 on ProcessBench")
print("=" * 70)

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load dataset
print("Loading ProcessBench dataset...")
dataset = load_dataset('Qwen/ProcessBench', split='gsm8k')
print(f"Dataset size: {len(dataset)} examples")

# Prepare evaluation data
print(f"\nPreparing {NUM_SAMPLES} evaluation samples...")
eval_texts = []
for i, item in enumerate(dataset):
    if i >= NUM_SAMPLES:
        break
    if 'problem' in item:
        text = item['problem']
    elif 'question' in item:
        text = item['question']
    elif 'text' in item:
        text = item['text']
    else:
        text = str(item)
    eval_texts.append(text)

# Tokenize all samples
print("Tokenizing...")
tokenized = tokenizer(
    eval_texts,
    padding='max_length',
    truncation=True,
    max_length=MAX_SEQ_LENGTH,
    return_tensors='pt'
)
input_ids = tokenized['input_ids']
attention_mask = tokenized['attention_mask']

print(f"Input shape: {input_ids.shape}")

# Cross-entropy loss function
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)


def evaluate_model(model, dtype_name):
    """Evaluate model and return average loss and time."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(0, NUM_SAMPLES, BATCH_SIZE):
            batch_ids = input_ids[i:i+BATCH_SIZE].to(DEVICE)
            batch_mask = attention_mask[i:i+BATCH_SIZE].to(DEVICE)
            
            # Forward pass
            outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
            logits = outputs.logits  # (batch, seq_len, vocab_size)
            
            # Compute loss: predict next token
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch_ids[:, 1:].contiguous()
            
            # Flatten for cross-entropy
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    elapsed_time = time.time() - start_time
    avg_loss = total_loss / num_batches
    
    return avg_loss, elapsed_time


# ===== BF16 Model =====
print("\n" + "=" * 70)
print("Loading Qwen3-0.6B in BFloat16...")
print("=" * 70)

model_bf16 = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE
)

print(f"Model dtype: {next(model_bf16.parameters()).dtype}")
print(f"Model device: {next(model_bf16.parameters()).device}")

# Evaluate bf16
print("\nEvaluating bf16 model...")
loss_bf16, time_bf16 = evaluate_model(model_bf16, "bf16")
print(f"BF16 - Loss: {loss_bf16:.6f}, Time: {time_bf16:.2f}s")

# Get memory usage
mem_bf16 = torch.cuda.max_memory_allocated(DEVICE) / (1024**3)
print(f"BF16 - Peak GPU memory: {mem_bf16:.2f} GB")

# Free bf16 model
del model_bf16
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(DEVICE)


# ===== Float32 Model =====
print("\n" + "=" * 70)
print("Loading Qwen3-0.6B in Float32...")
print("=" * 70)

model_fp32 = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map=DEVICE
)

print(f"Model dtype: {next(model_fp32.parameters()).dtype}")
print(f"Model device: {next(model_fp32.parameters()).device}")

# Evaluate fp32
print("\nEvaluating float32 model...")
loss_fp32, time_fp32 = evaluate_model(model_fp32, "fp32")
print(f"FP32 - Loss: {loss_fp32:.6f}, Time: {time_fp32:.2f}s")

# Get memory usage
mem_fp32 = torch.cuda.max_memory_allocated(DEVICE) / (1024**3)
print(f"FP32 - Peak GPU memory: {mem_fp32:.2f} GB")

# Free fp32 model
del model_fp32
torch.cuda.empty_cache()


# ===== Results Summary =====
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"\n{'Metric':<25} {'BFloat16':>15} {'Float32':>15} {'Difference':>15}")
print("-" * 70)
print(f"{'Cross-Entropy Loss':<25} {loss_bf16:>15.6f} {loss_fp32:>15.6f} {loss_bf16 - loss_fp32:>+15.6f}")
print(f"{'Evaluation Time (s)':<25} {time_bf16:>15.2f} {time_fp32:>15.2f} {time_bf16 - time_fp32:>+15.2f}")
print(f"{'Peak GPU Memory (GB)':<25} {mem_bf16:>15.2f} {mem_fp32:>15.2f} {mem_bf16 - mem_fp32:>+15.2f}")
print()

if loss_bf16 > loss_fp32:
    pct_diff = ((loss_bf16 - loss_fp32) / loss_fp32) * 100
    print(f"Float32 has {pct_diff:.2f}% LOWER loss than BFloat16")
else:
    pct_diff = ((loss_fp32 - loss_bf16) / loss_bf16) * 100
    print(f"BFloat16 has {pct_diff:.2f}% LOWER loss than Float32")

print(f"Float32 uses {(mem_fp32 / mem_bf16 - 1) * 100:.1f}% MORE GPU memory")
print(f"Float32 is {(time_fp32 / time_bf16 - 1) * 100:.1f}% {'SLOWER' if time_fp32 > time_bf16 else 'FASTER'}")
