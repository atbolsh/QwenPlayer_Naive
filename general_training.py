"""
General Training Script for QwenAgentPlayer

This script provides a flexible training loop that can train on multiple frameworks
with configurable options including LoRA support.

Usage:
    python general_training.py --use_lora --num_batches 10000 --batch_size 8
"""

import os
import random
import argparse
import warnings
from typing import List, Tuple, Callable, Optional

import torch
import torch.optim as optim

# Import frameworks package
from frameworks import (
    device, create_model, apply_lora,
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

# Checkpoint directory
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "brain_checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


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
    batch_size: int = 8,
    lr: float = 1e-5,
    use_lora: bool = False,
    checkpoint_prefix: str = "qwen_agent",
    save_every: int = 1000,
    print_every: int = 100,
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
    """
    # Apply LoRA if requested
    if use_lora:
        model = apply_lora(model)
    
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
    curr_mins = [1e6 for _ in frameworks]
    
    print(f"Starting training for {num_batches} batches...")
    print(f"Tasks: {task_names}")
    print(f"Repetition weights: {repetitions}")
    print(f"LoRA: {use_lora}")
    print("=" * 50)
    
    for b in range(num_batches):
        # Sample a task
        func, _, task_ind = rb.random_draw()
        batch_num = batch_counts[task_ind]
        batch_counts[task_ind] += 1
        
        # Reset model periodically to prevent memory issues
        reset_model = (b % 3 == 2)
        printing = ((batch_num % print_every) == (print_every - 1))
        
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
        except Exception as e:
            print(f"Error in task {task_names[task_ind]}: {e}")
            model.reset()
            continue
        
        # Print progress
        if printing:
            avg_loss = total_losses[task_ind] / print_every
            total_losses[task_ind] = 0
            print(f"Task {task_ind} ({task_names[task_ind]}), batch {batch_num}: avg loss = {avg_loss:.4f}")
            
            if avg_loss < curr_mins[task_ind]:
                curr_mins[task_ind] = avg_loss
                print(f"  -> New best for task {task_ind}!")
        
        # Save checkpoint
        if (b + 1) % save_every == 0:
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, 
                f"{checkpoint_prefix}_batch{b + 1}.pth"
            )
            torch.save(model.pipe.model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final checkpoint
    final_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_prefix}_final.pth")
    torch.save(model.pipe.model.state_dict(), final_path)
    print(f"Final checkpoint saved: {final_path}")
    
    return model


def get_default_frameworks() -> List[Tuple[Callable, int]]:
    """Get default framework configuration."""
    return [
        (arrow_task_batch, 8),
        (qa_task_batch, 8),
        (control_batch, 8),
        (mem_canvas_batch, 4),
        (blue_line_direction_batch, 4),
        (gold_direction_batch, 4),
        (gold_proximity_batch, 4),
        (please_turn_batch, 4),
        (relposition_qa_batch, 4),
        (direction_names_batch, 4),
        (zoom_task_batch, 2),
        (imagineWithoutYou_task_batch, 2),
        (imagineWithoutGold_task_batch, 2),
        (imagineWithoutWalls_task_batch, 2),
        (imagineWallsOnly_task_batch, 2),
        (imagineFacingGold_task_batch, 2),
        (imagineCloser2Gold_task_batch, 2),
        (imagineAfterMove_task_batch, 2),
    ]


def main():
    parser = argparse.ArgumentParser(description="Train QwenAgentPlayer")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA adapters")
    parser.add_argument("--num_batches", type=int, default=10000, help="Number of training batches")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--save_every", type=int, default=1000, help="Save checkpoint every N batches")
    parser.add_argument("--print_every", type=int, default=100, help="Print progress every N batches")
    parser.add_argument("--checkpoint_prefix", type=str, default="qwen_agent", help="Checkpoint filename prefix")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint to load")
    
    args = parser.parse_args()
    
    # Create model
    print("Creating model...")
    model = create_model(device=device, use_lora=False)  # LoRA applied in train()
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        print(f"Loading checkpoint: {args.load_checkpoint}")
        model.pipe.model.load_state_dict(torch.load(args.load_checkpoint, map_location=device))
    
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
