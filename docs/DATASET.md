# Dataset Configuration

## Primary Dataset: ProcessBench

The default training dataset is **Qwen's ProcessBench**:

```python
from datasets import load_dataset
import json

dataset = load_dataset('Qwen/ProcessBench', split='gsm8k')
print(json.dumps(dataset[0], indent=2))
```

ProcessBench contains mathematical reasoning problems, which helps the model develop structured reasoning capabilities.

## Using ProcessBench in Training

The `ProcessBenchDataset` class in `general_framework.py` wraps this dataset:

```python
from general_framework import ProcessBenchDataset

# Load dataset
dataset = ProcessBenchDataset(split='gsm8k', device='cuda')

# Get a batch
batch = dataset[0:32]
```

## Fallback: Local Text Files

If ProcessBench is unavailable, the framework falls back to local text files:

```
text_pretraining_data/
├── *-train.txt    # Training data
└── *-eval.txt     # Evaluation data
```

## Game-Specific Data

The game in `game/` generates visual environments. During training:

1. Game states are rendered as 224×224 images
2. Text descriptions/commands are tokenized with Qwen tokenizer
3. The model learns to:
   - Understand game state from images
   - Generate appropriate text responses
   - Reconstruct/predict images

## Custom Datasets

To create custom datasets for `QwenBastardBrain`:

```python
from torch.utils.data import Dataset
from general_framework import tokenizer, MAX_SEQ_LENGTH

class MyDataset(Dataset):
    def __init__(self, data_list, device='cpu'):
        self.device = device
        self.examples = []
        
        for text in data_list:
            encoded = tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors='pt'
            )
            self.examples.append(encoded['input_ids'].squeeze(0))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return self.examples[i].to(self.device)
```

## Important Notes

1. **Always use Qwen tokenizer** for encoding text (see `TOKENIZER.md`)
2. **Vocabulary size is 151936** — must match model configuration
3. **Sequence length** default is 32 tokens, adjustable via `MAX_SEQ_LENGTH`

