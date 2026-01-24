# API Reference

Complete API documentation for QwenAgentPlayer and related components.

## Table of Contents

1. [QwenAgentPlayer](#qwenagentplayer)
2. [QwenAgentPipe](#qwenagentpipe)
3. [QwenExtension](#qwenextension)
4. [Framework Functions](#framework-functions)
5. [Utility Functions](#utility-functions)
6. [Game Integration](#game-integration)

---

## QwenAgentPlayer

The main model class that wraps the complete multimodal agent.

**Location**: `visual_transformer/qwen_agent.py`

### Constructor

```python
QwenAgentPlayer(
    model_name: str = "Qwen/Qwen3-0.6B",
    device: torch.device = None,
    num_canvases: int = 3,
)
```

**Parameters:**
- `model_name`: HuggingFace model identifier for Qwen3
- `device`: Torch device (defaults to CUDA if available)
- `num_canvases`: Number of canvas images to keep in history

### Key Attributes

- `pipe`: `QwenAgentPipe` - The internal processing pipeline
- `canvases`: List of generated canvas images (most recent first)
- `device`: The device the model is on

### Methods

#### `reset()`
Clear all internal state including canvas history.

```python
model.reset()
```

#### `soft_reset()`
Clear gradients without clearing canvas history. Call after each training step.

```python
model.soft_reset()
```

#### `to(device)`
Move model to specified device.

```python
model.to(torch.device('cuda:1'))
```

---

## QwenAgentPipe

The internal pipeline handling tokenization, embedding, and inference.

### Methods

#### `forward(texts, images, generate_image=True)`

Run a forward pass through the model.

```python
result = model.pipe.forward(
    texts=["What do you see?"],
    images=image_tensor,  # Shape: (B, 3, 224, 224)
    generate_image=True,
)
```

**Parameters:**
- `texts`: List of text strings
- `images`: Tensor of shape (batch, 3, 224, 224)
- `generate_image`: Whether to decode the image

**Returns:** `PipeOutput` with:
- `logits`: Text prediction logits (seq_len, vocab_size)
- `generated_image`: Reconstructed image tensor (if requested)
- `hidden_states`: Internal hidden states

#### `batch_forward(input_ids, images, ...)`

Lower-level forward with pre-tokenized input.

```python
result = model.pipe.batch_forward(
    input_ids=token_ids,  # (batch, seq_len)
    images=image_tensor,
    generate_image=True,
)
```

---

## QwenExtension

Wraps a Qwen3 model with image encoding/decoding capabilities.

### Key Components

- `qwen_model`: The underlying Qwen3 model
- `img_enc`: Image encoder (converts 224x224x3 -> embeddings)
- `img_dec`: Image decoder (converts embeddings -> 224x224x3)

---

## Framework Functions

All framework batch functions share a common signature:

```python
def framework_batch(
    batch_size: int,
    model: QwenAgentPlayer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    batch_num: int = 0,
    compute_grad: bool = False,
    random_order: bool = True,
    model_eval: bool = True,
    reset_model: bool = True,
    printing: bool = True,
    training: bool = False,
    use_lora: bool = False,
) -> Tuple[float, ...]
```

**Parameters:**
- `batch_size`: Number of samples per batch
- `model`: QwenAgentPlayer instance
- `optimizer`: Optimizer for training (required if training=True)
- `batch_num`: Current batch number (for data cycling)
- `compute_grad`: Whether to compute gradients
- `random_order`: Shuffle task order within batch
- `model_eval`: Put model in eval mode
- `reset_model`: Reset model after batch
- `printing`: Print loss information
- `training`: Whether this is a training step (calls optimizer.step())
- `use_lora`: Whether using LoRA adapters

**Returns:** Tuple of loss values (total_loss, task_loss, control_loss, ...)

### Available Frameworks

```python
from frameworks import (
    # Core
    control_batch,
    arrow_task_batch,
    qa_task_batch,
    
    # QA tasks
    blue_line_direction_batch,
    gold_direction_batch,
    gold_proximity_batch,
    please_turn_batch,
    relposition_qa_batch,
    direction_names_batch,
    
    # Memory
    mem_canvas_batch,
    
    # Imagination
    imagineWithoutYou_task_batch,
    imagineWithoutGold_task_batch,
    imagineWithoutWalls_task_batch,
    imagineWallsOnly_task_batch,
    imagineFacingGold_task_batch,
    imagineCloser2Gold_task_batch,
    imagineAfterMove_task_batch,
    
    # Other
    zoom_task_batch,
    comparisonv1_task_batch,
    complex_loss_batch,
)
```

---

## Utility Functions

### Model Creation

```python
from frameworks import create_model, apply_lora

# Create model
model = create_model(
    model_name="Qwen/Qwen3-0.6B",
    device=torch.device('cuda'),
    use_lora=False,
)

# Apply LoRA to existing model
model = apply_lora(model, r=4, lora_alpha=16, lora_dropout=0.1)
```

### Text Encoding/Decoding

```python
from frameworks import tokenizer, encode_batch, decode_batch

# Encode text to token IDs
tokens = encode_batch(["Hello world", "Test"])  # Returns tensor

# Decode token IDs to text
texts = decode_batch(token_tensor)  # Returns list of strings
```

### Model Adapters

These functions bridge the old API to QwenAgentPlayer:

```python
from frameworks import model_forward, model_forward_with_tokens

# With string text (tokenizes internally)
text_probs, img_recon = model_forward(
    model, ["What do you see?"], img_tensor, ret_imgs=True
)

# With token ID tensors (PREFERRED - no encode/decode overhead)
# All frameworks use this for efficiency
text_probs, img_recon = model_forward_with_tokens(
    model, token_ids, img_tensor, ret_imgs=True
)
```

**Important:** `model_forward_with_tokens` operates directly on token tensors 
without any decode/encode round-trip. This is the preferred method for training
frameworks since they already work with tokenized data.

### Loss Functions

```python
from frameworks import img_criterion, get_text_loss

# Image reconstruction loss
img_loss = img_criterion(predicted_img, target_img)

# Text prediction loss
text_loss = get_text_loss(logits, target_tokens)
```

---

## Game Integration

### Creating Game States

```python
from frameworks import G, discreteGame, get_settings_batch, get_images

# Single game
settings = G.random_bare_settings(gameSize=224, max_agent_offset=0.5)
game = discreteGame(settings)

# Batch of games
settings_batch = get_settings_batch(batch_size=8)
images = get_images(settings_batch)  # Returns (B, 3, 224, 224) tensor
```

### Game Actions

```python
# Execute actions
game.actions[1]()  # Forward
game.actions[3]()  # Clockwise
game.actions[4]()  # Counter-clockwise
```

### Game Logic

```python
from frameworks import (
    gold_direction_angle,
    will_intersect_forward,
    should_turn_anticlockwise_forward,
    best_move_forward,
    trace_forward,
)

# Get angle to gold
angle = gold_direction_angle(game)

# Check if going forward will hit gold
will_hit = will_intersect_forward(game)

# Get best move (1=forward, 3=CW, 4=CCW)
best = best_move_forward(game)
```

---

## Checkpoint Management

### Saving

```python
import torch

# Save model state
torch.save(model.pipe.model.state_dict(), "checkpoint.pth")
```

### Loading

```python
# Load model state
model.pipe.model.load_state_dict(
    torch.load("checkpoint.pth", map_location=device)
)
```

---

## Error Handling

Common issues and solutions:

### CUDA Out of Memory
- Reduce batch size
- Use LoRA instead of full fine-tuning
- Reset model more frequently: `model.reset()`

### NaN Losses
- Check learning rate (try 1e-5 or lower)
- Ensure images are normalized correctly (0-1 range)
- Check for empty batches

### Import Errors
- Ensure you're in the project root directory
- Check that `game/` and `visual_transformer/` are accessible
