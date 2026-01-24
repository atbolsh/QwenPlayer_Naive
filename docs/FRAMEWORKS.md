# Training Frameworks Guide

This document describes all available training frameworks, their purposes, and how to use them effectively.

## Overview

Training frameworks are modular Python modules that define specific training tasks. Each framework:

1. Generates training data (text prompts, images, targets)
2. Runs the model forward pass
3. Computes task-specific losses
4. Optionally updates model parameters

## Framework Categories

### 1. Core Frameworks

#### `control.py` - Basic Control Task

**Purpose:** Basic image reconstruction and text prediction. This is the foundation that ensures the model can reconstruct input images and predict input text.

**Task:** Given an image and random text, reconstruct the image and predict the text.

**Usage:**
```python
from frameworks import control_batch

loss_info = control_batch(
    batch_size=8,
    model=model,
    optimizer=optimizer,
    training=True,
    compute_grad=True,
)
# Returns: (total_loss, img_loss, text_loss)
```

#### `arrow_to_gold.py` - Draw Arrow to Gold

**Purpose:** Train the model to imagine/draw a line from the agent to the gold.

**Task:** Given a game image and a prompt like "Draw the path to the gold", generate an image with an arrow pointing to the gold.

**Key learning:** Spatial understanding, goal-directed visualization.

### 2. Question-Answering Frameworks

These frameworks train the model to answer questions about the game state.

#### `position_qa.py` - Position Questions

**Purpose:** Answer questions about relative positions (left/right/up/down).

**Example Q&A:**
- "Is the gold to the left or right?" → "Left"
- "Are you above or below the gold?" → "Below"

#### `blue_line_qa.py` - Blue Line Direction

**Purpose:** Answer if the agent is facing the blue line direction.

#### `gold_direction_qa.py` - Gold Direction

**Purpose:** Answer if the agent is facing toward the gold.

#### `near_gold_qa.py` - Gold Proximity

**Purpose:** Answer if the agent is near the gold.

#### `relposition_qa.py` - Relative Position & Movement

**Purpose:** Answer complex questions about movement decisions.

**Example Q&A:**
- "If you go forward, will you hit gold?" → "Yes/No"
- "Which way should you turn?" → "Clockwise/Counter-clockwise"
- "What's the best move?" → "Forward/CW/CCW"

#### `please_turn_qa.py` - Turn Commands

**Purpose:** Respond to turn commands with correct action tokens.

**Example:**
- "Please turn towards the gold" → "<clock>" or "<anticlock>"

#### `direction_names.py` - Action Token Names

**Purpose:** Associate action tokens with their names.

**Example:**
- "What action is <forward>?" → "That's a move forward"
- "<clock> What did you do?" → "Clockwise turn"

### 3. Imagination Frameworks

These frameworks train the model to imagine modified versions of the current scene.

#### `imagine_without_you.py`
Imagine the scene without the agent present.

#### `imagine_without_gold.py`
Imagine the scene without the gold.

#### `imagine_without_walls.py`
Imagine the scene without walls.

#### `imagine_walls_only.py`
Imagine only the walls (no agent, no gold).

#### `imagine_facing_gold.py`
Imagine what the scene would look like if facing the gold.

#### `imagine_closer_to_gold.py`
Imagine being halfway to the gold.

#### `imagine_after_move.py`
Imagine the scene after a described sequence of moves.

**Example prompt:** "What would this look like after 3 forward steps?"

### 4. Memory Frameworks

#### `mem_canvas_use.py` - Canvas Memory

**Purpose:** Train the model to recall images from previous steps.

**Task:** After seeing a sequence of images, recall a specific one from history.

**Note:** Limited to 3 steps (matching `QwenAgentPlayer.num_canvases`).

**Example prompts:**
- "What did you see 1 image ago?"
- "Recall the current image"

### 5. Advanced Frameworks

#### `zoom.py` - Zoom Views

**Purpose:** Generate zoomed-in views of different parts of the scene.

**Subtasks:**
- `zoomAgent_task_batch`: Zoom on the agent
- `zoomGold_task_batch`: Zoom on the gold
- `zoomHalfway_task_batch`: Zoom on the path between

#### `comparison_v1.py` - State Comparison

**Purpose:** Compare two game states and choose the better one.

**Task:** Show two game images, ask which is "better" (closer to gold).

**Note:** This is a multi-step framework that requires seeing both images before answering.

#### `complex_loss_v1.py` - Complex Loss

**Purpose:** Train with differentiable loss based on detected positions.

Uses `image_to_settings.py` to extract agent/gold positions from generated images and applies losses based on:
- Agent moving closer to gold
- Maintaining correct radii
- Not moving the gold

---

## Creating Custom Frameworks

### Template

```python
# my_framework.py
from .general_framework import *
from .general_qa import *  # if using QA utilities

# Define prompts and responses
prompts = ["Prompt 1", "Prompt 2"]
prompts_tensor = tensorify_list(prompts)

def my_data_generator(batch_size):
    """Generate training data."""
    S = get_settings_batch(batch_size)
    imgs = get_images(S)
    texts = simple_sample(batch_size, prompts_tensor)
    # Generate targets...
    return imgs, texts, targets

def _my_batch(batch_size, model, optimizer=None, batch_num=0, 
              random_order=True, model_eval=True, reset_model=True, 
              printing=True, training=False, use_lora=False):
    """Internal batch function."""
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval both True")
    
    if model_eval:
        model.pipe.model.eval()
    if training:
        model.pipe.model.train()
    if training and optimizer is None:
        raise ValueError("Must provide optimizer for training")
    
    # Get data
    imgs, texts, targets = my_data_generator(batch_size)
    
    # Run model
    text_probs, img_recon = model_forward_with_tokens(
        model, texts, imgs, ret_imgs=True
    )
    
    # Compute losses
    img_loss = img_criterion(img_recon, targets)
    text_loss = get_text_loss(text_probs, texts)
    loss = img_loss + text_loss / 5000
    
    # Training step
    if training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.soft_reset()
    
    if printing:
        print(f"Loss: {loss.item()}")
    
    if reset_model:
        model.reset()
    
    return loss.item(), img_loss.item(), text_loss.item()

def my_batch(batch_size, model, optimizer=None, batch_num=0, 
             compute_grad=False, random_order=True, model_eval=True, 
             reset_model=True, printing=True, training=False, use_lora=False):
    """Public batch function with gradient control."""
    if compute_grad:
        return _my_batch(batch_size, model, optimizer, batch_num, 
                        random_order, model_eval, reset_model, 
                        printing, training, use_lora)
    else:
        if training:
            raise ValueError("training=True requires compute_grad=True")
        with torch.no_grad():
            return _my_batch(batch_size, model, optimizer, batch_num, 
                            random_order, model_eval, reset_model, 
                            printing, training, use_lora)
```

### Registering in `__init__.py`

Add to `frameworks/__init__.py`:

```python
from .my_framework import my_batch

__all__ = [
    # ... existing exports ...
    'my_batch',
]
```

---

## Framework Design Principles

1. **Always include control texts**: Run both task prompts and control prompts to prevent catastrophic forgetting.

2. **Random order**: Shuffle the order of different subtasks within a batch to prevent order-dependent learning.

3. **Use `model_forward_with_tokens`**: This adapter operates directly on token tensors without decode/encode overhead - more efficient for training.

4. **Call `model.soft_reset()` after training**: Clears gradients without clearing canvas history.

5. **Call `model.reset()` periodically**: Prevents memory accumulation; do this every few batches.

6. **Balance losses**: Text losses are typically much larger than image losses; divide by 5000 as a starting point.

7. **Use token tensors directly**: `model_forward_with_tokens` operates directly on token tensors without any decode/encode round-trip. This is more efficient for training.

---

## Training Tips

### Curriculum Learning
Start with simpler frameworks before complex ones:
1. `control_batch` (reconstruction)
2. `arrow_task_batch` (simple visualization)
3. `qa_task_batch` (position understanding)
4. More complex frameworks

### Framework Weights
In `general_training.py`, each framework has a weight for sampling frequency:

```python
frameworks = [
    (control_batch, 8),      # Run often
    (arrow_task_batch, 8),   # Run often
    (complex_loss_batch, 2), # Run less often
]
```

### Debugging
Set `printing=True` to see losses, and use small batch sizes initially:

```python
loss_info = my_batch(
    batch_size=2,
    model=model,
    printing=True,
    training=False,
)
```
