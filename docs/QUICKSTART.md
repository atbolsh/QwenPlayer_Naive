# QwenAgentPlayer Quickstart Guide

This guide gets you up and running with QwenAgentPlayer in 5 minutes.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd qwen-player

# Install dependencies
pip install -r requirements.txt

# Or run the setup script
bash setup.sh
```

## Quick Start

### 1. Basic Model Usage

```python
from frameworks import device, create_model, tokenizer

# Create model
model = create_model(device=device, use_lora=False)

# Generate with text and image
from frameworks import G, discreteGame
import torch

# Create a game state
game = discreteGame(G.random_bare_settings(gameSize=224))
img = torch.FloatTensor(game.getData()).unsqueeze(0)
img = torch.permute(img, (0, 3, 1, 2)).contiguous().to(device)

# Run forward pass
result = model.pipe.forward(
    texts=["What do you see?"],
    images=img,
    generate_image=True,
)

print(result.logits.shape)        # Text logits
print(result.generated_image.shape)  # Generated image
```

### 2. Training with Frameworks

```python
from frameworks import (
    device, create_model, 
    control_batch, arrow_task_batch
)
import torch.optim as optim

# Create model
model = create_model(device=device, use_lora=False)

# Create optimizer
optimizer = optim.Adam(model.pipe.model.parameters(), lr=1e-5)

# Train on a batch
loss_info = arrow_task_batch(
    batch_size=8,
    model=model,
    optimizer=optimizer,
    batch_num=0,
    compute_grad=True,
    training=True,
    printing=True,
)
```

### 3. Using LoRA for Efficient Fine-tuning

```python
# Create model with LoRA adapters
model = create_model(device=device, use_lora=True)

# Only LoRA parameters are trainable
# ~1% of total parameters
```

### 4. Running the Training Scripts

```bash
# Standard training
python general_training.py --num_batches 10000 --batch_size 8

# Training with LoRA
python general_training.py --use_lora --num_batches 10000

# Parallel training (multiple frameworks per step)
python general_parallel_training.py --num_parallel 4 --use_lora

# Load from checkpoint
python general_training.py --load_checkpoint brain_checkpoints/model.pth
```

### 5. Interactive Exploration

```bash
# Run interactive demo
python demo_interactive.py

# Or run the basic demo
python demo_qwen_agent.py
```

For Jupyter notebooks, see `Widget_Interface_RoughDraft.ipynb`.

## Available Frameworks

| Framework | Description |
|-----------|-------------|
| `control_batch` | Basic image reconstruction |
| `arrow_task_batch` | Draw line to gold |
| `qa_task_batch` | Position Q&A |
| `mem_canvas_batch` | Image recall |
| `blue_line_direction_batch` | Blue line direction Q&A |
| `gold_direction_batch` | Gold direction Q&A |
| `please_turn_batch` | Turn commands |
| `zoom_task_batch` | Zoom views |
| `imagine*_task_batch` | Various imagination tasks |

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Read [API_REFERENCE.md](API_REFERENCE.md) for detailed API docs
- Read [FRAMEWORKS.md](FRAMEWORKS.md) for framework details
- Read [CLAUDE_GUIDE.md](CLAUDE_GUIDE.md) if you're an AI assistant
