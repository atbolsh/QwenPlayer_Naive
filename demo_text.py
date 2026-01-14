"""Demo: Text processing with QwenBastardBrain's sentence_autoencoder"""

import torch
from general_framework import (
    QwenBastardBrain, device, tokenizer, sdt,
    encode_text, decode_text
)

# Load model
model = QwenBastardBrain().to(device)
model.eval()

# Get a batch from ProcessBench dataset
batch_size = 4
text_batch = torch.stack([sdt[i] for i in range(batch_size)]).to(device)

print("=== Input Texts ===")
for i in range(batch_size):
    print(f"{i}: {decode_text(text_batch[i], skip_special_tokens=True)[:80]}...")

# Run sentence autoencoder (no context, just text reconstruction)
with torch.no_grad():
    logits = model.sentence_autoencoder(text_batch, context=None, return_full=True, use_masks=True)

# Get predictions (argmax over vocab dimension)
predictions = torch.argmax(logits, dim=1)  # batch x seq_len

print("\n=== Reconstructed Texts ===")
for i in range(batch_size):
    print(f"{i}: {decode_text(predictions[i], skip_special_tokens=True)[:80]}...")

# Show prediction quality
print("\n=== Token Accuracy ===")
for i in range(batch_size):
    matches = (predictions[i] == text_batch[i]).float().mean()
    print(f"Sample {i}: {matches.item()*100:.1f}% tokens match")

