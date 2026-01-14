# Tokenizer Configuration

## Required Tokenizer

**Always use the Qwen tokenizer** for `QwenBastardBrain`:

```python
from transformers import AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

## Key Properties

- **Vocabulary size**: 151936
- **Pad token ID**: `tokenizer.pad_token_id`
- **EOS token ID**: `tokenizer.eos_token_id`
- **BOS token ID**: `tokenizer.bos_token_id`

## Special Tokens for Game Control

The framework adds custom special tokens for game actions:

```python
SPECIAL_TOKENS = ['<forward>', '<clock>', '<anticlock>']
tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
```

**Important**: After adding special tokens, if you're using the full Qwen model (not just the bastard encoder/decoder), you may need to resize embeddings:
```python
model.resize_token_embeddings(len(tokenizer))
```

## Encoding Text

Use the helper functions from `general_framework.py`:

```python
from general_framework import encode_text, encode_batch, decode_text, decode_batch

# Single text
token_ids = encode_text("Hello world", max_length=32)

# Batch of texts
token_ids = encode_batch(["Hello", "World"], max_length=32)

# Decode back
text = decode_text(token_ids, skip_special_tokens=True)
texts = decode_batch(token_ids_batch, skip_special_tokens=True)
```

## Legacy Tokenizer (Do Not Use)

The old `EnhancedAgentBrain` used a custom ByteLevelBPE tokenizer:
```python
# OLD - DO NOT USE WITH QwenBastardBrain
tokenizer = ByteLevelBPETokenizer(
    "./text_pretraining_tokenizer/eng_sentences_tokenizer_vc10000_v2-vocab.json",
    "./text_pretraining_tokenizer/eng_sentences_tokenizer_vc10000_v2-merges.txt",
)
```

This tokenizer has vocabulary size 10000 and is **incompatible** with `QwenBastardBrain`.

## Max Sequence Length

Default: **32 tokens**

This is set via `MAX_SEQ_LENGTH` in `general_framework.py`. Adjust as needed for your use case.

