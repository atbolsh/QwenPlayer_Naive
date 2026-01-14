# Data Types and Precision

## Overview

`QwenBastardBrain` uses **mixed precision** with careful dtype management:

| Component | Data Type | Dimension |
|-----------|-----------|-----------|
| Qwen Encoder | bfloat16 | 1024 |
| Qwen Decoder | bfloat16 | 1024 |
| Image Encoder | float32 | 1024 |
| Image Decoder | float32 | 1024 |
| Memory | float32 | 1024 |
| Dopamine | float32 | 1024 |
| Context Tagging | float32 | 1024 |

## Why bfloat16 for Qwen?

The Qwen3 model is trained and distributed in bfloat16. Using bfloat16:
- Reduces memory usage by ~50%
- Maintains numerical stability (unlike float16)
- Preserves pretrained weights' precision

## Conversion Points

### Text Encoding
```python
def get_text_encoding(self, text_batch, ...):
    # Qwen encoder outputs bfloat16
    encoding = self.text_enc(input_ids=text_batch)
    # Convert to float32 for rest of network
    return encoding.float()
```

### Text Decoding
```python
def get_text_decoding(self, text_encoding, ..., context=None, ...):
    # Convert to bfloat16 for Qwen decoder
    text_encoding_bf16 = text_encoding.to(torch.bfloat16)
    if context is not None:
        context_bf16 = context.to(torch.bfloat16)
    
    # Qwen decoder operates in bfloat16
    logits = self.text_dec(input_embeds=text_encoding_bf16, context=context_bf16)
    
    # Convert back to float32
    return logits.float()
```

## Consistent Embedding Dimension

**All components use 1024 embedding dimension.**

This is set in `__init__`:
```python
embed_dim = 1024

self.img_enc = ImageTransformerEncoder(embed_dim=embed_dim)
self.img_dec = ImageTransformerDecoder(embed_dim=embed_dim)
self.img_weight = VisionWeightedSum(embed_dim=embed_dim)
self.dopamine = DopamineWrapper(embed_dim=embed_dim)
self.memory = Memory(mem_size, new_tokens, vector_dim=embed_dim)
self.mem_enc = MemoryEncoder(new_tokens=new_tokens, embed_dim=embed_dim)
```

## Comparison with EnhancedAgentBrain

| Property | EnhancedAgentBrain | QwenBastardBrain |
|----------|-------------------|------------------|
| Embedding dim | 768 | 1024 |
| Text encoder | SentenceTransformerEncoder (float32) | Qwen3_BastardEncoder (bfloat16) |
| Text decoder | SentenceTransformerDecoder (float32) | Qwen3_BastardDecoder (bfloat16) |
| Vocab size | 10000 | 151936 |
| All components | float32 | Mixed (bfloat16/float32) |

## Best Practices

1. **Don't mix dtypes carelessly** — always use explicit `.float()` or `.to(torch.bfloat16)` at boundaries
2. **Loss computation in float32** — always convert to float32 before computing losses
3. **Optimizer state in float32** — standard practice for mixed precision training
4. **Check device AND dtype** when debugging numerical issues

