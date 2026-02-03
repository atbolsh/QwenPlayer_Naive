"""Measure per-layer INPUT magnitudes for Qwen3-0.6B.

This captures the typical scale of vectors entering each transformer layer,
which is what we need for scaling image context in load_bases.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
load_dotenv()

device = 'cuda:0'

print("Loading Qwen3-0.6B...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    torch_dtype=torch.bfloat16,
).to(device)
model.eval()

# Sample texts for measuring typical magnitudes
sample_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models process text tokens efficiently.",
    "What is the meaning of life and the universe?",
    "Python programming language is widely used for data science.",
    "The capital of France is Paris, a beautiful city.",
    "Artificial intelligence is transforming many industries.",
    "Mathematics provides the foundation for computer science.",
    "Climate change is one of the biggest challenges we face.",
]

print(f"\nMeasuring per-layer INPUT magnitudes across {len(sample_texts)} samples...")
print("="*70)

# Storage for magnitudes per layer
layer_input_magnitudes = {i: [] for i in range(28)}
embedding_magnitudes = []

# Hook to capture layer inputs
layer_inputs = {}

def make_hook(layer_idx):
    def hook(module, input, output):
        # input is a tuple, first element is hidden_states
        hidden_states = input[0]
        layer_inputs[layer_idx] = hidden_states.detach()
    return hook

# Register hooks on each decoder layer
hooks = []
for i, layer in enumerate(model.model.layers):
    h = layer.register_forward_hook(make_hook(i))
    hooks.append(h)

# Process each text
with torch.no_grad():
    for text in sample_texts:
        tokens = tokenizer(text, return_tensors='pt')['input_ids'].to(device)
        
        # Get embedding output (input to layer 0)
        embeddings = model.model.embed_tokens(tokens)
        emb_mag = torch.norm(embeddings, dim=-1).mean().item()
        embedding_magnitudes.append(emb_mag)
        
        # Forward pass triggers all hooks
        layer_inputs.clear()
        _ = model(tokens)
        
        # Collect magnitudes from each layer's input
        for layer_idx in range(28):
            if layer_idx in layer_inputs:
                inp = layer_inputs[layer_idx]
                mag = torch.norm(inp, dim=-1).mean().item()
                layer_input_magnitudes[layer_idx].append(mag)

# Remove hooks
for h in hooks:
    h.remove()

# Compute averages
avg_embedding_mag = np.mean(embedding_magnitudes)
avg_layer_mags = [np.mean(layer_input_magnitudes[i]) for i in range(28)]

print(f"\nEmbedding layer output (input to transformer): {avg_embedding_mag:.4f}")
print("\nPer-layer INPUT magnitudes:")
print("-"*50)
for i, mag in enumerate(avg_layer_mags):
    ratio = mag / avg_embedding_mag
    print(f"Layer {i:2d} input: {mag:8.4f}  (ratio to embedding: {ratio:6.2f}x)")

# Also get img_enc magnitude for reference
print("\n" + "="*70)
print("For reference - Image encoder output magnitude: ~31.73")
print("(from previous measurement)")

# Print as Python list for easy copy-paste
print("\n" + "="*70)
print("PYTHON CONSTANTS FOR COPY-PASTE:")
print("="*70)
print(f"\nEMBEDDING_MAGNITUDE = {avg_embedding_mag:.4f}")
print(f"IMG_ENC_MAGNITUDE = 31.73  # from previous measurement")
print(f"\n# Per-layer input magnitudes (layers 0-27)")
print("LAYER_INPUT_MAGNITUDES = [")
for i, mag in enumerate(avg_layer_mags):
    comma = "," if i < 27 else ""
    print(f"    {mag:.4f}{comma}  # Layer {i}")
print("]")

# Scaling factors for image context
print(f"\n# Scaling factors: layer_mag / img_enc_mag")
print("LAYER_SCALE_FACTORS = [")
img_enc_mag = 31.73
for i, mag in enumerate(avg_layer_mags):
    scale = mag / img_enc_mag
    comma = "," if i < 27 else ""
    print(f"    {scale:.6f}{comma}  # Layer {i}: {mag:.2f} / {img_enc_mag}")
print("]")

print("\nDone!")
