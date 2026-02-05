# VRAM Usage Analysis for Training Frameworks

**Date**: January 2026  
**Context**: Investigation into why batch size dropped from 75 to 12 when adding more frameworks.

---

## Summary

When training with multiple frameworks, VRAM usage increases dramatically due to:
1. Multiple forward passes per framework (activation memory accumulation)
2. Module-level GPU tensors created at import time
3. KV-cache accumulation across forward passes

---

## Forward Passes Per Framework

PyTorch retains ALL intermediate activations from forward passes until `backward()` is called. Frameworks that do multiple forward passes before backward require proportionally more memory.

| Framework | Forward Passes | File |
|-----------|---------------|------|
| `control_batch` | 1 | `control.py` |
| `arrow_task_batch` | 2 | `arrow_to_gold.py` |
| `qa_task_batch` | **5** | `position_qa.py` |
| `mem_canvas_batch` | 3+ | `mem_canvas_use.py` |
| `zoom_task_batch` | 2 | `zoom.py` |
| `imagine*_task_batch` | 2 | `imagine_*.py` |
| `blue_line_direction_batch` | varies | `blue_line_qa.py` |
| `gold_direction_batch` | varies | `gold_direction_qa.py` |
| `gold_proximity_batch` | varies | `near_gold_qa.py` |
| `please_turn_batch` | varies | `please_turn_qa.py` |
| `relposition_qa_batch` | varies | `relposition_qa.py` |
| `direction_names_batch` | varies | `direction_names.py` |

**Rule of thumb**: If batch_size=75 works for control (1 pass), then:
- 2-pass framework → ~batch_size 37
- 5-pass framework → ~batch_size 15

---

## Import-Time GPU Tensors

Each framework creates tensors at module import time that stay in GPU memory permanently. Even if you only USE 3 frameworks, ALL frameworks get imported via the `from .general_framework import *` chain.

| Framework File | Tensors Created at Import |
|----------------|---------------------------|
| `arrow_to_gold.py` | 1 (`task1_text_tensor`) |
| `zoom.py` | 3 |
| `please_turn_qa.py` | 6 |
| `relposition_qa.py` | 9 |
| `position_qa.py` | 12 |
| `comparison_v1.py` | 4 |
| `complex_loss_v1.py` | 1 |
| `blue_line_qa.py` | 3 |
| `near_gold_qa.py` | 3 |
| `gold_direction_qa.py` | 3 |
| `direction_names.py` | 1 |
| `imagine_*.py` | 1 each |

These are created via `tensorify_list()` or `encode_batch().to(device)` at the top of each file.

---

## Potential Solutions (Not Yet Implemented)

### 1. Gradient Checkpointing
Trade compute for memory - recompute activations during backward pass instead of storing them.

### 2. Sequential Backward Passes
Instead of:
```python
out1 = forward(...)  # stores activations
out2 = forward(...)  # stores MORE activations
loss = f(out1, out2)
loss.backward()      # needs ALL activations
```

Do:
```python
out1 = forward(...)
loss1 = f(out1)
loss1.backward()     # release activations immediately
optimizer.step(); optimizer.zero_grad()

out2 = forward(...)
loss2 = f(out2)
loss2.backward()
optimizer.step(); optimizer.zero_grad()
```

### 3. Lazy Tensor Loading
Move import-time tensors to CPU, load to GPU on-demand within each batch function.

### 4. Restructure Frameworks
Redesign multi-pass frameworks to be single-pass where possible.

---

## Practical Workaround

If framework X does N forward passes and framework Y does M forward passes, weight them in the framework list proportionally:

```python
# In get_default_frameworks():
return [
    (control_batch, 8 * 3),  # 1 pass, weight 3x to balance with QA's 5 passes
    (arrow_task_batch, 8),    # 2 passes
    (qa_task_batch, 8),       # 5 passes
]
```

This doesn't reduce VRAM, but ensures balanced training across tasks.

---

## Baseline Reference

**Untrained Qwen3-0.6B text loss**: ~3.0 (cross-entropy), perplexity ~20  
See `baseline_text_loss.txt` for details.
