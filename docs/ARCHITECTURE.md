# QwenBastardBrain Architecture

## Overview

`QwenBastardBrain` is a multimodal neural network architecture that combines:
- **Qwen3 language model components** for text understanding and generation
- **Vision Transformers** for image processing
- **Memory systems** for context retention
- **Dopamine-like reward signals** for reinforcement learning

The end goal of this project is to create an agent that can both **play the game** defined in the `game/` folder AND **talk intelligently about it** — reasoning about game states, explaining strategies, and responding to natural language queries.

## Key Components

### Text Processing (1024-dim, bfloat16)
- `Qwen3_BastardEncoder`: First half of Qwen3-0.6B model layers for encoding text to embeddings
- `Qwen3_BastardDecoder`: Second half of Qwen3-0.6B model layers for generating text from embeddings
- Vocabulary size: **151936** (Qwen's full vocabulary)

### Vision Processing (1024-dim, float32)
- `ImageTransformerEncoder`: Encodes 224x224 images to 256 patches × 1024-dim embeddings
- `ImageTransformerDecoder`: Reconstructs images from embeddings
- `VisionWeightedSum`: Computes weighted combinations of multiple image features
- `VisionCanvases`: Stores 3 most recent generated images

### Memory System (1024-dim, float32)
- `Memory`: Ring-buffer style storage for 128 memory tokens
- `MemoryEncoder`: Compresses text+context into memory tokens

### Reward Signal
- `DopamineWrapper`: Produces a scalar reward signal from image/text context

## Data Type Handling

**Critical**: The Qwen encoder/decoder operate in **bfloat16** while the rest of the network uses **float32**.

Conversions happen at these boundaries:
1. `get_text_encoding()`: Qwen encoder output (bfloat16) → converted to float32
2. `get_text_decoding()`: Input converted to bfloat16 → Qwen decoder → output converted back to float32

All components use **1024 embedding dimension** consistently.

## Context Structure

The model builds context from 7 sources (all 1024-dim):
1. Current input image encoding
2. Canvas 0 (most recent generated image)
3. Canvas 1
4. Canvas 2
5. Memory tensor
6. Dopamine/reward signal
7. Text encoding

Each source has a learned `context_tagging` vector added to distinguish it.

## HuggingFace Integration

`QwenBastardBrain` inherits from `PyTorchModelHubMixin` for easy upload/download:

```python
# Save to HuggingFace
model.push_to_hub("your-username/qwen-bastard-brain")

# Load from HuggingFace
model = QwenBastardBrain.from_pretrained("your-username/qwen-bastard-brain")
```

## File Structure

```
visual_transformer/
├── qwen_player.py       # QwenBastardBrain class
├── qwen_encoders.py     # Qwen3_BastardEncoder/Decoder
├── enhanced_model.py    # Original EnhancedAgentBrain (768-dim)
├── model.py             # ImageTransformer, SentenceTransformer components
├── memory.py            # Memory, MemoryEncoder, MemoryProcessor
├── vision_canvas.py     # VisionCanvases
└── custom_transformer.py # PositionalEncoding, PatchEmbedding, etc.
```

