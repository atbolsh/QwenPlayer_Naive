# QwenAgentPlayer Architecture

## Overview

**QwenAgentPlayer** is a multimodal neural network architecture that combines:
- **Qwen3 language model** for text understanding and generation
- **Image encoder/decoder** for visual processing
- **Canvas history** for tracking generated images

The goal is to create an agent that can:
1. **Play the game** defined in `game/`
2. **Talk about it** - reasoning about game states, explaining strategies
3. **Imagine modifications** - visualize hypothetical scenarios

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      QwenAgentPlayer                             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                     QwenAgentPipe                           ││
│  │  ┌─────────────┐  ┌──────────────────────────────────────┐ ││
│  │  │  Tokenizer  │  │          QwenExtension               │ ││
│  │  │  (HF)       │  │  ┌──────────────────────────────┐   │ ││
│  │  └─────────────┘  │  │        Qwen3 Model           │   │ ││
│  │                   │  │    (Language Processing)      │   │ ││
│  │                   │  └──────────────────────────────┘   │ ││
│  │                   │  ┌─────────────┐ ┌─────────────┐   │ ││
│  │                   │  │  img_enc    │ │  img_dec    │   │ ││
│  │                   │  │  (Encoder)  │ │  (Decoder)  │   │ ││
│  │                   │  └─────────────┘ └─────────────┘   │ ││
│  │                   └──────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Canvas History (3 images)                      ││
│  │  [Canvas 0: Most Recent] [Canvas 1] [Canvas 2: Oldest]     ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### QwenAgentPlayer

**Location:** `visual_transformer/qwen_agent.py`

The top-level wrapper that manages:
- The processing pipeline (`pipe`)
- Canvas history for generated images
- Device management
- State reset methods

```python
class QwenAgentPlayer:
    def __init__(self, model_name="Qwen/Qwen3-0.6B", device=None, num_canvases=3):
        self.pipe = QwenAgentPipe(model_name, device)
        self.canvases = []  # Most recent first
        self.num_canvases = num_canvases
```

### QwenAgentPipe

Handles the processing pipeline:
1. Tokenizes input text
2. Embeds images using `img_enc`
3. Creates combined context
4. Runs through Qwen3 model
5. Decodes image output using `img_dec`

```python
class QwenAgentPipe:
    def forward(self, texts, images, generate_image=True):
        # 1. Tokenize text
        tokens = self.tokenize(texts)
        
        # 2. Embed image
        img_embeddings = self.model.img_enc(images)
        
        # 3. Combine and process
        context = self.create_context(tokens, img_embeddings)
        output = self.model.qwen_model(context)
        
        # 4. Generate image
        if generate_image:
            generated = self.model.img_dec(output)
        
        return PipeOutput(logits=output.logits, generated_image=generated)
```

### QwenExtension

Extends the base Qwen3 model with image capabilities:

```python
class QwenExtension(nn.Module):
    def __init__(self, model_name):
        self.qwen_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.img_enc = ImageTransformerEncoder(...)  # 224x224x3 → embeddings
        self.img_dec = ImageTransformerDecoder(...)  # embeddings → 224x224x3
```

## Data Flow

### Input Processing

```
Text Input: "Draw the path to the gold"
    ↓ Tokenizer
Token IDs: [256, 1234, 567, ...]
    ↓ Embedding Layer
Text Embeddings: (seq_len, embed_dim)

Image Input: (batch, 3, 224, 224)
    ↓ Patch Embedding + Positional Encoding
    ↓ Transformer Layers
Image Embeddings: (num_patches, embed_dim)
```

### Combined Processing

```
[Image Embeddings] + [Text Embeddings]
    ↓ Concatenate
Combined Context: (total_len, embed_dim)
    ↓ Qwen3 Transformer Layers
Output: (total_len, embed_dim)
    ↓
Text Output: Logits for next token prediction
Image Output: Reconstructed/modified image
```

### Canvas Management

```
After each generation:
    new_image = img_dec(output)
    canvases.insert(0, new_image)  # Add to front
    if len(canvases) > 3:
        canvases.pop()  # Remove oldest
```

## Embedding Dimensions

| Component | Dimension | Notes |
|-----------|-----------|-------|
| Qwen3 embeddings | 1024 | For Qwen3-0.6B |
| Image patches | 256 | 14×14 grid + CLS token |
| Each patch | 1024 | Matches Qwen |
| Vocabulary size | 151936 | Qwen tokenizer |

## Model Sizes

| Variant | Parameters | VRAM (Inference) | VRAM (Training) |
|---------|------------|------------------|-----------------|
| Qwen3-0.6B | ~600M | ~2GB | ~8GB |
| With LoRA | +~4M trainable | ~2GB | ~4GB |

## State Management

### soft_reset()
Called after each training batch:
- Clears gradients
- Keeps canvas history
- Preserves model state

### reset()
Called between episodes:
- Clears gradients
- Clears canvas history
- Returns model to initial state

## Training Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Loop                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                 Framework Selection                         ││
│  │  [control] [arrow] [qa] [imagine] [zoom] [compare] ...     ││
│  └─────────────────────────────────────────────────────────────┘│
│                            ↓                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  Batch Generation                           ││
│  │  - Random game states                                       ││
│  │  - Task-specific prompts                                    ││
│  │  - Target outputs (images/text)                             ││
│  └─────────────────────────────────────────────────────────────┘│
│                            ↓                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  Forward Pass                               ││
│  │  model_forward_with_tokens(model, tokens, images)           ││
│  └─────────────────────────────────────────────────────────────┘│
│                            ↓                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  Loss Computation                           ││
│  │  loss = img_criterion(recon, target) + text_loss/5000       ││
│  └─────────────────────────────────────────────────────────────┘│
│                            ↓                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  Backward + Optimize                        ││
│  │  loss.backward() → optimizer.step() → model.soft_reset()    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## LoRA Integration

When using LoRA (Low-Rank Adaptation):

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=4,                    # Rank
    lora_alpha=16,          # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Attention layers
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model.pipe.model.qwen_model = get_peft_model(
    model.pipe.model.qwen_model, 
    lora_config
)
```

Benefits:
- ~1% of parameters trainable
- Faster training
- Lower memory usage
- Easy to swap adapters

## File Organization

```
visual_transformer/
├── qwen_agent.py        # QwenAgentPlayer, QwenAgentPipe, QwenExtension
├── model.py             # ImageTransformerEncoder/Decoder, SentenceTransformer
├── custom_transformer.py # PositionalEncoding, PatchEmbedding
├── vision_canvas.py     # VisionCanvases (legacy)
├── memory.py            # Memory (legacy, not used in QwenAgentPlayer)
└── enhanced_model.py    # EnhancedAgentBrain (legacy, replaced)
```

## Historical Context

### Previous Architecture: EnhancedAgentBrain

The previous system used:
- Separate `Qwen3_BastardEncoder` and `Qwen3_BastardDecoder`
- Explicit `Memory` objects with ring buffers
- `VisionCanvases` with learned weights
- `DopamineWrapper` for reward signals

### Current Architecture: QwenAgentPlayer

Simplified design:
- Unified `QwenExtension` wrapping full Qwen3
- Simple list-based canvas history
- No separate memory system (relies on attention)
- No dopamine (to be added differently later)

### Migration

Adapter functions in `frameworks/general_framework.py` bridge old code:

```python
# Old API (frameworks still use this via adapters)
model(text_ids, images, ret_imgs=True)

# New API - token tensors (efficient, no decode/encode overhead)
model.batch_forward(input_ids=token_ids, image=img_tensor, generate_image=True)

# New API - string input (convenient but tokenizes internally)
model.pipe.forward(text=["Hello"], images=[img], generate_image=True)
```

The `model_forward_with_tokens` adapter operates directly on token tensors,
calling `model.batch_forward()` without any wasteful decode/encode round-trips.

## Future Directions

1. **Memory augmentation**: Add explicit memory without the complexity of the old system
2. **Reward learning**: Reintroduce dopamine-like signals for RL
3. **Multi-step planning**: Extend canvas history for longer horizons
4. **Action prediction**: Direct action token generation for game playing
