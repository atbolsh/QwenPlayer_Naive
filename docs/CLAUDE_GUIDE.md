# Claude AI Assistant Guide

This document is specifically written for AI assistants (Claude, GPT, etc.) who will work on this codebase. It provides essential context, patterns, and pitfalls to avoid.

## Project Overview

**QwenAgentPlayer** is a multimodal AI agent that:
- Takes text prompts and game images as input
- Generates text responses and modified images as output
- Maintains a history of generated images ("canvases")
- Is trained on various task-specific "frameworks"

## Architecture Summary

```
QwenAgentPlayer (main model wrapper)
├── pipe: QwenAgentPipe (processing pipeline)
│   ├── tokenizer: HuggingFace tokenizer
│   └── model: QwenExtension
│       ├── qwen_model: Qwen3 LLM
│       ├── img_enc: Image encoder (224x224x3 → embeddings)
│       └── img_dec: Image decoder (embeddings → 224x224x3)
├── canvases: List[Tensor] (generated image history, max 3)
└── device: torch.device
```

## Key Files

| File/Directory | Purpose |
|----------------|---------|
| `visual_transformer/qwen_agent.py` | Main model classes |
| `frameworks/` | Training task modules |
| `frameworks/general_framework.py` | Core utilities, model adapters, tokenizer |
| `frameworks/general_qa.py` | QA-specific utilities |
| `frameworks/game_logic_solver.py` | Game movement/direction logic |
| `game/discreteEngine.py` | Game environment |
| `general_training.py` | Training script |

## Critical Patterns

### 1. Model Forward Pass

**CORRECT (for frameworks - uses token tensors directly):**
```python
from frameworks import model_forward_with_tokens

# When you have token IDs (most common in frameworks)
# This operates directly on tensors - no decode/encode overhead
text_probs, img_recon = model_forward_with_tokens(
    model, token_ids, img_tensor, ret_imgs=True
)
```

**ALSO CORRECT (for string inputs):**
```python
from frameworks import model_forward

# For string inputs (tokenizes internally)
text_probs, img_recon = model_forward(
    model, ["What do you see?"], img_tensor, ret_imgs=True
)
```

**LOWER LEVEL (direct model access):**
```python
# Use batch_forward for direct tensor operations
result = model.batch_forward(
    input_ids=token_ids,
    image=img_tensor,
    attention_mask=attention_mask,
    generate_image=True,
)
```

**WRONG:**
```python
# Don't decode tokens then re-encode - wasteful!
texts = tokenizer.batch_decode(tokens)  # BAD
encoded = tokenizer(texts)  # Wasteful round-trip
```

### 2. Model State Management

**IMPORTANT:** Always manage model state properly:

```python
# After each training step:
model.soft_reset()  # Clears gradients, keeps canvases

# After each episode/periodically:
model.reset()  # Clears everything including canvases
```

### 3. Framework Batch Function Signature

All framework batch functions MUST follow this signature:

```python
def my_batch(
    batch_size: int,
    model,
    optimizer=None,
    batch_num: int = 0,
    compute_grad: bool = False,
    random_order: bool = True,
    model_eval: bool = True,
    reset_model: bool = True,
    printing: bool = True,
    training: bool = False,
    use_lora: bool = False,
) -> Tuple[float, ...]:
```

### 4. Import Patterns in Frameworks

**Within frameworks/ directory, use relative imports:**
```python
from .general_framework import *
from .general_qa import *
from .game_logic_solver import some_function
```

**From outside frameworks/, use absolute imports:**
```python
from frameworks import create_model, control_batch
```

### 5. Image Tensor Format

```python
# Standard format: (batch, channels, height, width)
# Values: 0-1 float
img_tensor.shape  # (B, 3, 224, 224)

# Converting from numpy game data:
img = torch.FloatTensor(game.getData())  # (224, 224, 3)
img = img.unsqueeze(0)                    # (1, 224, 224, 3)
img = torch.permute(img, (0, 3, 1, 2))    # (1, 3, 224, 224)
img = img.contiguous().to(device)
```

### 6. Text Tokenization

```python
from frameworks import tokenizer, encode_batch, decode_batch

# Encode strings to token tensors
token_ids = encode_batch(["text1", "text2"])  # Returns (B, seq_len) tensor

# Decode token tensors to strings
strings = decode_batch(token_ids)  # Returns list of strings
```

## Common Pitfalls

### 1. Wrong Model Eval/Train Mode

```python
# WRONG: Using model.eval()/model.train() directly
model.eval()  # Won't work correctly

# CORRECT: Access the inner model
model.pipe.model.eval()
model.pipe.model.train()
```

### 2. Forgetting to Reset Model

```python
# Memory will accumulate and cause OOM
for batch in batches:
    train_batch(model)  # BAD: Never resets

# CORRECT: Reset periodically
for i, batch in enumerate(batches):
    train_batch(model)
    if i % 3 == 2:
        model.reset()
```

### 3. Wrong Loss Balance

```python
# Image loss is ~0.01-0.1, text loss is ~100-1000
# WRONG: Equal weighting
loss = img_loss + text_loss  # Text dominates

# CORRECT: Scale text loss down
loss = img_loss + (text_loss / 5000)
```

### 4. Tensor Device Mismatch

```python
# Always ensure tensors are on the right device
from frameworks import device
tensor = tensor.to(device)
```

### 5. Forgetting sys.path for Game Imports

If creating new files in `frameworks/`, they need access to `game/`:

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game import discreteGame
```

## Adding New Features

### New Training Framework

1. Create `frameworks/my_framework.py`
2. Import from `.general_framework` and `.general_qa`
3. Follow the batch function signature pattern
4. Add to `frameworks/__init__.py`
5. Add to the default frameworks in `general_training.py`

### New Model Component

1. Add to `visual_transformer/qwen_agent.py`
2. Update `QwenAgentPlayer.__init__`
3. Ensure `reset()` and `soft_reset()` handle new state
4. Update `visual_transformer/__init__.py` exports

### New Game Feature

1. Add to `game/discreteEngine.py`
2. If needed for training, add helper to `frameworks/game_logic_solver.py`

## Testing Changes

```python
# Quick test for a framework
from frameworks import device, create_model, my_batch

model = create_model(device=device)
loss = my_batch(batch_size=2, model=model, printing=True)
print(f"Test passed, loss: {loss}")
```

## File Organization

```
qwen-player/
├── frameworks/           # Training frameworks (EDIT HERE MOST OFTEN)
│   ├── __init__.py      # Exports all framework functions
│   ├── general_framework.py  # Core utilities
│   ├── general_qa.py    # QA utilities
│   ├── game_logic_solver.py  # Game logic
│   └── *.py             # Individual frameworks
├── visual_transformer/   # Model code (RARELY EDIT)
│   ├── qwen_agent.py    # Main model classes
│   └── ...
├── game/                 # Game environment (RARELY EDIT)
├── docs/                 # Documentation
├── general_training.py   # Training script
└── demo_*.py            # Demo scripts
```

## Debugging Tips

1. **Check shapes first:**
   ```python
   print(f"img: {img.shape}, text: {text.shape}")
   ```

2. **Use small batch sizes:**
   ```python
   result = my_batch(batch_size=1, printing=True)
   ```

3. **Test without gradients first:**
   ```python
   with torch.no_grad():
       result = my_batch(batch_size=2, compute_grad=False)
   ```

4. **Check for NaN:**
   ```python
   if torch.isnan(loss):
       print("NaN detected!")
   ```

## Historical Context

This project evolved from an earlier `EnhancedAgentBrain` system that used:
- Separate encoder/decoder classes (`Qwen3_BastardEncoder`, `Qwen3_BastardDecoder`)
- Memory objects for long-term storage
- Different API (`model(text, img, ret_imgs=True)`)

The new `QwenAgentPlayer` simplifies this with:
- Unified `QwenExtension` model
- Canvas-based image history (no separate memory)
- Pipeline-based API (`model.pipe.forward(...)`)

Adapter functions in `general_framework.py` (`model_forward`, `model_forward_with_tokens`) bridge the old framework code to the new API.

## Questions to Ask the User

When unclear, ask about:
1. Which GPU/device to use
2. Whether to use LoRA or full fine-tuning
3. Which frameworks to prioritize
4. Checkpoint paths
5. Batch sizes (memory constraints)
