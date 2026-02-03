"""
Demo Script: QwenAgentPlayer Basic Usage

This script demonstrates how to use the QwenAgentPlayer model for:
1. Creating and initializing the model
2. Running forward passes with text and images
3. Training on frameworks
4. Using LoRA for efficient fine-tuning

Run with: python demo_qwen_agent.py
"""

import torch
import matplotlib.pyplot as plt

# ==============================================================================
# SECTION 1: Model Creation
# ==============================================================================

print("=" * 60)
print("SECTION 1: Creating QwenAgentPlayer")
print("=" * 60)

from frameworks import device, create_model, apply_lora_to_text, tokenizer

# Create the model (this loads Qwen3 and initializes image encoder/decoder)
print(f"\nUsing device: {device}")
print("Creating model... (this may take a moment)")

model = create_model(device=device, use_lora=False)
print(f"Model created successfully!")
print(f"Model type: {type(model).__name__}")

# ==============================================================================
# SECTION 2: Basic Forward Pass
# ==============================================================================

print("\n" + "=" * 60)
print("SECTION 2: Basic Forward Pass")
print("=" * 60)

from frameworks import G, discreteGame, get_settings_batch, get_images

# Create a random game state
settings = G.random_bare_settings(gameSize=224, max_agent_offset=0.5)
game = discreteGame(settings)

# Get image as tensor
img_data = torch.FloatTensor(game.getData()).unsqueeze(0)
img_tensor = torch.permute(img_data, (0, 3, 1, 2)).contiguous().to(device)

print(f"\nImage tensor shape: {img_tensor.shape}")

# Run forward pass with text
text_prompt = "What do you see in this game?"
print(f"Text prompt: '{text_prompt}'")

# Method 1: Using pipe.forward with strings (convenient)
result = model.pipe.forward(
    text=[text_prompt],
    images=[img_tensor[0]],  # List of images
    generate_image=True,
)

print(f"\nForward pass complete!")
print(f"Logits shape: {result['outputs'].logits.shape}")
if result.get('generated_image') is not None:
    print(f"Generated image shape: {result['generated_image'].shape}")

# Method 2: Using model_forward_with_tokens (efficient - no decode/encode overhead)
from frameworks import encode_batch, model_forward_with_tokens

token_ids = encode_batch([text_prompt]).to(device)
text_probs, img_recon = model_forward_with_tokens(model, token_ids, img_tensor, ret_imgs=True)
print(f"\nToken-based forward pass complete!")
print(f"Text probs shape: {text_probs.shape}  # (batch, vocab, seq_len)")
print(f"Image recon shape: {img_recon.shape}")

# ==============================================================================
# SECTION 3: Using Framework Batch Functions
# ==============================================================================

print("\n" + "=" * 60)
print("SECTION 3: Using Framework Batch Functions")
print("=" * 60)

from frameworks import control_batch, arrow_task_batch, qa_task_batch

# Run a control batch (no training, just evaluation)
print("\nRunning control_batch (evaluation mode)...")
model.reset()

loss_info = control_batch(
    batch_size=4,
    model=model,
    optimizer=None,
    batch_num=0,
    compute_grad=False,
    random_order=True,
    model_eval=True,
    reset_model=True,
    printing=True,
    training=False,
)

print(f"Control batch losses: {loss_info}")

# ==============================================================================
# SECTION 4: Training with an Optimizer
# ==============================================================================

print("\n" + "=" * 60)
print("SECTION 4: Training Example")
print("=" * 60)

import torch.optim as optim

# Create optimizer
optimizer = optim.Adam(model.pipe.model.parameters(), lr=1e-5, eps=1e-9)

# Run a single training batch
print("\nRunning arrow_task_batch (training mode)...")
model.reset()

loss_info = arrow_task_batch(
    batch_size=4,
    model=model,
    optimizer=optimizer,
    batch_num=0,
    compute_grad=True,
    random_order=True,
    model_eval=False,
    reset_model=True,
    printing=True,
    training=True,
)

print(f"Arrow task losses: {loss_info}")

# ==============================================================================
# SECTION 5: Using LoRA
# ==============================================================================

print("\n" + "=" * 60)
print("SECTION 5: LoRA Fine-tuning")
print("=" * 60)

# Create a fresh model with LoRA
print("\nCreating model with LoRA adapters...")
model_lora = create_model(device=device, use_lora=True)

# Count trainable parameters
total_params = sum(p.numel() for p in model_lora.pipe.model.parameters())
trainable_params = sum(p.numel() for p in model_lora.pipe.model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters (LoRA): {trainable_params:,}")
print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

# ==============================================================================
# SECTION 6: Saving and Loading Checkpoints
# ==============================================================================

print("\n" + "=" * 60)
print("SECTION 6: Checkpoints")
print("=" * 60)

import os

checkpoint_dir = "brain_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Save checkpoint
checkpoint_path = os.path.join(checkpoint_dir, "demo_checkpoint.pth")
torch.save(model.pipe.model.state_dict(), checkpoint_path)
print(f"Checkpoint saved to: {checkpoint_path}")

# Load checkpoint (strict=False for compatibility with different model versions)
model.pipe.model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
print(f"Checkpoint loaded successfully!")

# Clean up demo checkpoint
os.remove(checkpoint_path)
print(f"Demo checkpoint cleaned up.")

# ==============================================================================
# SECTION 7: Available Frameworks
# ==============================================================================

print("\n" + "=" * 60)
print("SECTION 7: Available Frameworks")
print("=" * 60)

framework_list = [
    ("control_batch", "Basic image reconstruction and text prediction"),
    ("arrow_task_batch", "Draw line from agent to gold"),
    ("qa_task_batch", "Answer position questions (left/right/up/down)"),
    ("mem_canvas_batch", "Recall images from history (3 steps)"),
    ("blue_line_direction_batch", "Answer if facing the blue line"),
    ("gold_direction_batch", "Answer if facing the gold"),
    ("gold_proximity_batch", "Answer if near the gold"),
    ("please_turn_batch", "Turn towards/away from targets"),
    ("relposition_qa_batch", "Relative position and movement questions"),
    ("direction_names_batch", "Learn action token names"),
    ("zoom_task_batch", "Generate zoomed views"),
    ("comparisonv1_task_batch", "Compare two game states"),
    ("complex_loss_batch", "Move agent closer to gold"),
    ("imagineWithoutYou_task_batch", "Imagine room without agent"),
    ("imagineWithoutGold_task_batch", "Imagine room without gold"),
    ("imagineWithoutWalls_task_batch", "Imagine room without walls"),
    ("imagineWallsOnly_task_batch", "Imagine only walls"),
    ("imagineFacingGold_task_batch", "Imagine facing the gold"),
    ("imagineCloser2Gold_task_batch", "Imagine being closer to gold"),
    ("imagineAfterMove_task_batch", "Imagine after a sequence of moves"),
]

print("\nAvailable training frameworks:\n")
for name, description in framework_list:
    print(f"  {name}")
    print(f"    -> {description}\n")

print("=" * 60)
print("Demo complete!")
print("=" * 60)
