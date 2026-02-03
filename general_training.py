"""
General Training Script for QwenAgentPlayer

This script provides a flexible training loop that can train on multiple frameworks
with configurable options including LoRA support.

Usage:
    python general_training.py --use_lora --num_batches 10000 --batch_size 8
"""

import os
import csv
import random
import argparse
import warnings
from typing import List, Tuple, Callable, Optional

import torch
import torch.optim as optim
from torchvision.utils import save_image

# Import frameworks package
from frameworks import (
    device, create_model, apply_lora_to_text, tokenizer,
    model_forward_with_tokens, encode_batch,
    G, get_images,
    control_batch, arrow_task_batch, qa_task_batch,
    mem_canvas_batch, blue_line_direction_batch,
    gold_direction_batch, gold_proximity_batch,
    please_turn_batch, relposition_qa_batch,
    direction_names_batch, zoom_task_batch,
    comparisonv1_task_batch, complex_loss_batch,
    imagineWithoutYou_task_batch, imagineWithoutGold_task_batch,
    imagineWithoutWalls_task_batch, imagineWallsOnly_task_batch,
    imagineFacingGold_task_batch, imagineCloser2Gold_task_batch,
    imagineAfterMove_task_batch,
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================
# EASILY EDITABLE: Default checkpoint to load at startup
# Set to None to use the default from frameworks/general_framework.py
# Or set to a path like "brain_checkpoints/your_checkpoint.pt"
# NOTE: The default loading happens in frameworks/general_framework.py
#       at import time (FRANKENSTEIN_CHECKPOINT_BF16 variable).
#       This override is applied via --load_checkpoint argument.
# ============================================================
DEFAULT_INIT_CHECKPOINT = "brain_checkpoints/frankenstein_finetune_control_better_embeddings_bf16.pt"

# Directories
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "brain_checkpoints")
DEMO_DIR = os.path.join(os.path.dirname(__file__), "demo_images")
LEDGER_PATH = os.path.join(os.path.dirname(__file__), "training_losses.csv")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DEMO_DIR, exist_ok=True)


def save_demo_images(model, step: int, task_name: str, prompt: str = "What do you see?"):
    """
    Save sample input/output images for a given task.
    
    Args:
        model: QwenAgentPlayer instance
        step: Current training step (for filename)
        task_name: Name of the task (for filename)
        prompt: Text prompt to use
    """
    model.pipe.model.eval()
    with torch.no_grad():
        # Generate a game image (bf16)
        settings = G.random_bare_settings(gameSize=224)
        img = get_images([settings], device=device)
        
        # Run model forward
        tokens = encode_batch([prompt]).to(device)
        _, img_recon = model_forward_with_tokens(model, tokens, img, ret_imgs=True)
        
        # Save images (convert to float32 for torchvision)
        safe_task_name = task_name.replace("_batch", "").replace("_task", "")
        input_path = os.path.join(DEMO_DIR, f"step_{step:06d}_{safe_task_name}_input.png")
        output_path = os.path.join(DEMO_DIR, f"step_{step:06d}_{safe_task_name}_output.png")
        
        save_image(img[0].float(), input_path)
        if img_recon is not None:
            save_image(img_recon[0].float().clamp(0, 1), output_path)
        
    model.pipe.model.train()
    model.reset()
    return input_path, output_path


class ReusableBuffer:
    """
    Buffer for random task sampling with repetition control.
    
    Allows different tasks to be sampled with different frequencies.
    """
    def __init__(self, L: List, repetitions: List[int]):
        self.L = []
        self.true_inds = []
        for i in range(len(L)):
            for j in range(repetitions[i]):
                self.L.append(L[i])
                self.true_inds.append(i)
        self.inds = list(range(len(self.L)))

    def draw(self, ind: int):
        return self.L[ind]

    def random_draw(self) -> Tuple:
        """Draw a random task, removing it from the pool temporarily."""
        ind_ind = random.randint(0, len(self.inds) - 1)
        ind = self.inds[ind_ind]
        if ind_ind == (len(self.inds) - 1):
            self.inds = self.inds[:-1]
        else:
            self.inds = self.inds[:ind_ind] + self.inds[ind_ind + 1:]
        if len(self.inds) == 0:
            self.inds = list(range(len(self.L)))
        return self.L[ind], ind, self.true_inds[ind]


def train(
    model,
    frameworks: List[Tuple[Callable, int]],
    num_batches: int = 10000,
    batch_size: int = 24, #8,
    lr: float = 1e-5,
    use_lora: bool = False,
    checkpoint_prefix: str = "qwen_agent",
    save_every: int = 1000,
    print_every: int = 100,
    ledger_path: str = LEDGER_PATH,
):
    """
    Train QwenAgentPlayer on multiple frameworks.
    
    Args:
        model: QwenAgentPlayer instance
        frameworks: List of (batch_func, repetition_weight) tuples
        num_batches: Total number of batches to train
        batch_size: Batch size for training
        lr: Learning rate
        use_lora: Whether to use LoRA adapters
        checkpoint_prefix: Prefix for checkpoint filenames
        save_every: Save checkpoint every N batches
        print_every: Print progress every N batches
        ledger_path: Path to save loss CSV
    """
    # Apply LoRA if requested
    if use_lora:
        model = apply_lora_to_text(model)
    
    # Create optimizer
    optimizer = optim.Adam(model.pipe.model.parameters(), lr=lr, eps=1e-9)
    
    # Create reusable buffer for task sampling
    batch_funcs = [f[0] for f in frameworks]
    repetitions = [f[1] for f in frameworks]
    rb = ReusableBuffer(batch_funcs, repetitions)
    
    # Track losses per task
    task_names = [f[0].__name__ for f in frameworks]
    batch_counts = [0 for _ in frameworks]
    total_losses = [0.0 for _ in frameworks]
    loss_counts = [0 for _ in frameworks]  # Track how many batches contributed to total_losses
    curr_mins = [1e6 for _ in frameworks]
    last_losses = [None for _ in frameworks]  # Track most recent loss for each task
    
    # Initialize CSV ledger with columns for each framework
    with open(ledger_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['global_batch'] + task_names
        writer.writerow(header)
    
    print(f"Starting training for {num_batches} batches...")
    print(f"Tasks: {task_names}")
    print(f"Repetition weights: {repetitions}")
    print(f"LoRA: {use_lora}")
    print(f"Loss ledger: {ledger_path}")
    print("=" * 50)
    
    for b in range(num_batches):
        # Sample a task
        func, _, task_ind = rb.random_draw()
        batch_num = batch_counts[task_ind]
        batch_counts[task_ind] += 1
        
        # Reset model periodically to prevent memory issues
        reset_model = (b % 3 == 2)
        
        # Print/log every print_every steps OR for the first 10 steps (debugging)
        should_print = ((b + 1) % print_every == 0) or (b < 10)
        printing = should_print
        
        # Run training batch
        try:
            full_results = func(
                batch_size, 
                model, 
                optimizer=optimizer, 
                batch_num=batch_num, 
                compute_grad=True, 
                random_order=True, 
                model_eval=False, 
                reset_model=reset_model, 
                printing=printing, 
                training=True,
                use_lora=use_lora,
            )
            L = full_results[0]  # Total loss
            total_losses[task_ind] += L
            loss_counts[task_ind] += 1
            last_losses[task_ind] = L
        except Exception as e:
            print(f"Error in task {task_names[task_ind]}: {e}")
            model.reset()
            continue
        
        # Print progress and log to CSV
        if should_print:
            # Calculate average loss for the current task
            if loss_counts[task_ind] > 0:
                avg_loss = total_losses[task_ind] / loss_counts[task_ind]
            else:
                avg_loss = L
            
            print(f"Batch {b + 1}/{num_batches} | Task {task_ind} ({task_names[task_ind]}), "
                  f"task_batch {batch_num}: loss = {L:.4f}, avg = {avg_loss:.4f}")
            
            if avg_loss < curr_mins[task_ind]:
                curr_mins[task_ind] = avg_loss
                print(f"  -> New best for task {task_ind}!")
            
            # Log to CSV - record average loss for ALL frameworks
            # Use 'inf' for frameworks that haven't been sampled yet
            row = [b + 1]
            for i in range(len(frameworks)):
                if loss_counts[i] > 0:
                    row.append(f"{total_losses[i] / loss_counts[i]:.6f}")
                else:
                    row.append('inf')
            with open(ledger_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            # Reset accumulated losses after print_every global batches
            if (b + 1) % print_every == 0:
                total_losses = [0.0 for _ in frameworks]
                loss_counts = [0 for _ in frameworks]
        
        # Save checkpoint and demo images
        if (b + 1) % save_every == 0:
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, 
                f"{checkpoint_prefix}_batch{b + 1}.pth"
            )
            torch.save(model.pipe.model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            
            # Reset canvases before generating demo images to avoid issues
            model.reset()
            
            # Save demo images for a few representative tasks
            demo_tasks = [
                (control_batch, "control", "What do you see?"),
                # (arrow_task_batch, "arrow", "Draw the path to the gold."),
                # (imagineWithoutYou_task_batch, "imagine_no_you", "Show me the room without yourself."),
            ]
            for demo_func, demo_name, demo_prompt in demo_tasks:
                try:
                    input_path, output_path = save_demo_images(model, b + 1, demo_name, demo_prompt)
                    print(f"Demo images saved: {demo_name}")
                except Exception as e:
                    print(f"Error saving demo images for {demo_name}: {e}")
            
            # Reset canvases after demo images
            model.reset()
    
    # Save final checkpoint
    final_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_prefix}_final.pth")
    torch.save(model.pipe.model.state_dict(), final_path)
    print(f"Final checkpoint saved: {final_path}")
    
    # Reset canvases before generating final demo images
    model.reset()
    
    # Save final demo images
    print("Saving final demo images...")
    demo_tasks = [
        (control_batch, "control", "What do you see?"),
        # (arrow_task_batch, "arrow", "Draw the path to the gold."),
        # (imagineWithoutYou_task_batch, "imagine_no_you", "Show me the room without yourself."),
    ]
    for demo_func, demo_name, demo_prompt in demo_tasks:
        try:
            save_demo_images(model, num_batches, f"final_{demo_name}", demo_prompt)
        except Exception as e:
            print(f"Error saving final demo images for {demo_name}: {e}")
    
    # Reset canvases after demo images
    model.reset()
    
    return model


def get_default_frameworks() -> List[Tuple[Callable, int]]:
    """Get default framework configuration."""
    return [
        # Only training on control_batch for now
        (control_batch, 8),
        # (arrow_task_batch, 8),
        # (qa_task_batch, 8),
        # (mem_canvas_batch, 4),
        # (blue_line_direction_batch, 4),
        # (gold_direction_batch, 4),
        # (gold_proximity_batch, 4),
        # (please_turn_batch, 4),
        # (relposition_qa_batch, 4),
        # (direction_names_batch, 4),
        # (zoom_task_batch, 2),
        # (imagineWithoutYou_task_batch, 2),
        # (imagineWithoutGold_task_batch, 2),
        # (imagineWithoutWalls_task_batch, 2),
        # (imagineWallsOnly_task_batch, 2),
        # (imagineFacingGold_task_batch, 2),
        # (imagineCloser2Gold_task_batch, 2),
        # (imagineAfterMove_task_batch, 2),
    ]


def main():
    parser = argparse.ArgumentParser(description="Train QwenAgentPlayer")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA adapters")
    parser.add_argument("--num_batches", type=int, default=10000000000, help="Number of training batches")
    parser.add_argument("--batch_size", type=int, default=70, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--save_every", type=int, default=1000, help="Save checkpoint every N batches")
    parser.add_argument("--print_every", type=int, default=100, help="Print progress every N batches")
    parser.add_argument("--checkpoint_prefix", type=str, default="qwen_agent_finetuned_vision_better_embeddings", help="Checkpoint filename prefix")
    parser.add_argument("--load_checkpoint", type=str, default=DEFAULT_INIT_CHECKPOINT, 
                        help="Path to checkpoint to load (default: DEFAULT_INIT_CHECKPOINT at top of file)")
    
    args = parser.parse_args()
    
    # Create model
    print("Creating model...")
    model = create_model(device=device, use_lora=False)  # LoRA applied in train()
    
    # Load checkpoint if specified (use --load_checkpoint "" to skip)
    # NOTE: Model already has default frankenstein weights from frameworks/general_framework.py import
    if args.load_checkpoint and args.load_checkpoint.strip():
        checkpoint_path = args.load_checkpoint
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            # Use strict=False to allow loading old checkpoints without layer_scale_factors
            model.pipe.model.load_state_dict(
                torch.load(checkpoint_path, map_location=device, weights_only=True),
                strict=False
            )
        else:
            print(f"WARNING: Checkpoint not found: {checkpoint_path}")
            print("Using weights loaded at import time from frameworks/general_framework.py")
    
    # Get default frameworks
    frameworks = get_default_frameworks()
    
    # Train
    train(
        model=model,
        frameworks=frameworks,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        lr=args.lr,
        use_lora=args.use_lora,
        checkpoint_prefix=args.checkpoint_prefix,
        save_every=args.save_every,
        print_every=args.print_every,
    )


if __name__ == "__main__":
    main()
