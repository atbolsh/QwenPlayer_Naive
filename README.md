# Qwen Player

A multimodal neural network using Qwen3 for game-playing agents. The model can understand game states through images, respond to natural language prompts, and generate modified visualizations.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Create directories
bash setup.sh

# Run demo
python demo_qwen_agent.py
```

## Usage

### Basic Model Usage

```python
from frameworks import device, create_model, tokenizer

# Create model (optionally with LoRA)
model = create_model(device=device, use_lora=False)

# Generate with text and image
from frameworks import G, discreteGame
import torch

game = discreteGame(G.random_bare_settings(gameSize=224))
img = torch.FloatTensor(game.getData()).unsqueeze(0)
img = torch.permute(img, (0, 3, 1, 2)).contiguous().to(device)

result = model.pipe.forward(
    texts=["Draw the path to the gold"],
    images=img,
    generate_image=True,
)
```

### Training

```bash
# Standard training on multiple frameworks
python general_training.py --num_batches 10000 --batch_size 8

# Training with LoRA (efficient fine-tuning)
python general_training.py --use_lora --num_batches 10000

# Load from checkpoint
python general_training.py --load_checkpoint brain_checkpoints/model.pth
```

### Interactive Exploration

```bash
# Terminal-based demo
python demo_interactive.py

# Or use the Jupyter widget
jupyter notebook Widget_Interface_RoughDraft.ipynb
```

## Project Structure

```
qwen-player/
├── frameworks/              # Training framework modules
│   ├── general_framework.py # Core utilities
│   ├── control.py          # Basic reconstruction task
│   ├── arrow_to_gold.py    # Draw arrow to gold
│   ├── position_qa.py      # Position Q&A
│   └── ...                 # 20+ frameworks
├── visual_transformer/      # Model code
│   ├── qwen_agent.py       # QwenAgentPlayer (main model)
│   └── ...
├── game/                    # Game environment
├── docs/                    # Documentation
├── general_training.py      # Main training script
├── demo_qwen_agent.py       # Basic demo
└── demo_interactive.py      # Interactive demo
```

## Available Frameworks

| Category | Frameworks |
|----------|------------|
| Core | `control_batch`, `arrow_task_batch` |
| Q&A | `qa_task_batch`, `blue_line_direction_batch`, `gold_direction_batch`, `gold_proximity_batch`, `please_turn_batch`, `relposition_qa_batch`, `direction_names_batch` |
| Memory | `mem_canvas_batch` |
| Imagination | `imagineWithoutYou_task_batch`, `imagineWithoutGold_task_batch`, `imagineFacingGold_task_batch`, ... |
| Other | `zoom_task_batch`, `comparisonv1_task_batch`, `complex_loss_batch` |

## Documentation

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](docs/QUICKSTART.md) | Get started in 5 minutes |
| [API_REFERENCE.md](docs/API_REFERENCE.md) | Complete API documentation |
| [FRAMEWORKS.md](docs/FRAMEWORKS.md) | Training framework details |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Model architecture |
| [CLAUDE_GUIDE.md](docs/CLAUDE_GUIDE.md) | Guide for AI assistants |
| [PROJECT_GOALS.md](docs/PROJECT_GOALS.md) | Project objectives |

## Model Architecture

**QwenAgentPlayer** combines:
- **Qwen3 LLM**: Text understanding and generation
- **Image Encoder**: 224×224 images → embeddings
- **Image Decoder**: Embeddings → 224×224 images
- **Canvas History**: 3 most recent generated images

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- ~4GB VRAM for inference, ~16GB for training

## Legacy Training Scripts

These scripts use older training approaches:

```bash
python train_naked_image.py   # Image autoencoder only
python train_both.py          # Text + Image with LoRA
python train_image.py         # Image autoencoder
python train_text_lora.py     # Text with LoRA
```

## License

[Add license information]

## Citation

[Add citation information]
