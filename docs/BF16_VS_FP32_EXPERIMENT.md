# BFloat16 vs Float32 Experiment

## Purpose

Investigate whether using float32 instead of bfloat16 could improve training outcomes. The concern was that bf16's limited precision may cause gradients to be too small to update weights effectively during training.

## Background

Qwen3-0.6B is distributed in bfloat16 and was likely finetuned for quantization. The question was whether the pretrained weights would perform worse in float32 (since they weren't optimized for that precision).

## Experiment

**Script:** `full_float_experiment.py`

Loaded Qwen3-0.6B in both bf16 and fp32, then measured cross-entropy loss on 100 samples from the ProcessBench gsm8k dataset.

## Results

| Metric | BFloat16 | Float32 | Difference |
|--------|----------|---------|------------|
| Cross-Entropy Loss | 2.455 | 2.458 | -0.003 |
| GPU Memory | 1.66 GB | 3.32 GB | +1.66 GB |
| Eval Time | 6.99s | 4.78s | -2.21s |

## Key Findings

1. **Loss is essentially identical** - Only 0.1% difference, with bf16 slightly better. This confirms the pretrained weights work equally well in both precisions.

2. **Float32 is viable for training** - The model doesn't degrade when converted to fp32, meaning we can use float32 for training to get better gradient precision without worrying about the pretrained weights being "incompatible."

3. **Memory tradeoff** - Float32 uses 2x GPU memory, which limits batch size but may be worth it for better gradient updates.

## Implications for Training

The issue with bf16 training is not inference quality but **gradient precision during backpropagation**:
- Small gradients in bf16 may round to zero, preventing weight updates
- Float32 provides better precision for accumulating small gradient updates
- This is especially important for fine-tuning where changes are incremental

## Recommendation

For training tasks where gradients are small (like the arrow-to-gold task), consider:
1. Using float32 for the entire model
2. Or using mixed precision with fp32 gradients/optimizer states
