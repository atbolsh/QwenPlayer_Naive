# Qwen Player

A multimodal neural network using Qwen3 encoder/decoder for game-playing agents.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create necessary directories:
   ```bash
   bash setup.sh
   ```

## Training

### Naked Image Autoencoder (lightweight, no Qwen model)
```bash
python train_naked_image.py
```

### Full Model Training
```bash
python train_both.py      # Text + Image jointly with LoRA
python train_image.py     # Image autoencoder with full model
python train_text_lora.py # Text with LoRA
```

## Documentation

See the `docs/` folder for detailed documentation:
- `ARCHITECTURE.md` - Model architecture
- `PROJECT_GOALS.md` - Project objectives
- `DATASET.md` - Dataset information
- `DATATYPES.md` - Data type specifications
- `TOKENIZER.md` - Tokenizer details
