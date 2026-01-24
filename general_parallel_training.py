"""
General Parallel Training Script for QwenAgentPlayer

This script provides data-parallel training where multiple frameworks
process different batches simultaneously on the same model instance.

Usage:
    python general_parallel_training.py --use_lora --num_batches 10000 --num_parallel 4
"""

import os
import argparse
import warnings
from typing import List, Tuple, Callable

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
    comparisonv1_task_batch,
    imagineWithoutYou_task_batch, imagineWithoutGold_task_batch,
    imagineWithoutWalls_task_batch, imagineWallsOnly_task_batch,
    imagineFacingGold_task_batch, imagineCloser2Gold_task_batch,
    imagineAfterMove_task_batch,
)

from parallel_frameworks import (
    ParallelFrameworkRunner,
    create_parallel_batches,
    run_parallel_training_step,
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Checkpoint directory
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "brain_checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_default_frameworks() -> List[Tuple[Callable, int]]:
    """Get default framework configuration with weights."""
    return [
        # Core tasks - higher weight
        (arrow_task_batch, 8),
        (qa_task_batch, 8),
        (control_batch, 8),
        
        # QA tasks - medium weight
        (blue_line_direction_batch, 4),
        (gold_direction_batch, 4),
        (gold_proximity_batch, 4),
        (please_turn_batch, 4),
        (relposition_qa_batch, 4),
        (direction_names_batch, 4),
        
        # Multi-step tasks
        (mem_canvas_batch, 4),
        (comparisonv1_task_batch, 2),
        
        # Imagination tasks - lower weight
        (zoom_task_batch, 2),
        (imagineWithoutYou_task_batch, 2),
        (imagineWithoutGold_task_batch, 2),
        (imagineWithoutWalls_task_batch, 2),
        (imagineWallsOnly_task_batch, 2),
        (imagineFacingGold_task_batch, 2),
        (imagineCloser2Gold_task_batch, 2),
        (imagineAfterMove_task_batch, 2),
    ]


def train_parallel(
    model,
    frameworks: List[Tuple[Callable, int]],
    num_batches: int = 10000,
    batch_size: int = 8,
    num_parallel: int = 4,
    lr: float = 1e-5,
    use_lora: bool = False,
    checkpoint_prefix: str = "qwen_agent_parallel",
    save_every: int = 1000,
    print_every: int = 100,
):
    """
    Train QwenAgentPlayer with data-parallel framework execution.
    
    Args:
        model: QwenAgentPlayer instance
        frameworks: List of (batch_func, weight) tuples
        num_batches: Total number of parallel training steps
        batch_size: Batch size per framework
        num_parallel: Number of frameworks to run per step
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
    
    # Create framework configs
    multi_step_frameworks = ['mem_canvas_batch', 'comparisonv1_task_batch']
    framework_configs = create_parallel_batches(frameworks, multi_step_frameworks)
    
    # Create parallel runner
    runner = ParallelFrameworkRunner(model, framework_configs, device)
    
    # Dry run to initialize all frameworks
    runner.dry_run_frameworks(batch_size=min(batch_size, 4))
    
    print(f"\nStarting parallel training for {num_batches} steps...")
    print(f"Frameworks: {[f.name for f in framework_configs]}")
    print(f"Parallel batch size: {num_parallel} frameworks per step")
    print(f"LoRA: {use_lora}")
    print("=" * 60)
    
    # Track losses
    running_loss = 0.0
    best_loss = float('inf')
    
    for step in range(num_batches):
        # Run parallel training step
        results = run_parallel_training_step(
            runner=runner,
            batch_size=batch_size,
            optimizer=optimizer,
            num_parallel=num_parallel,
            use_lora=use_lora,
        )
        
        step_loss = results.get("total_loss", 0.0)
        running_loss += step_loss
        
        # Print progress
        if (step + 1) % print_every == 0:
            avg_loss = running_loss / print_every
            running_loss = 0.0
            
            print(f"Step {step + 1}/{num_batches}: avg loss = {avg_loss:.4f}")
            
            # Print per-framework losses
            for name, info in results.items():
                if name != "total_loss" and isinstance(info, dict):
                    loss = info.get("loss", "N/A")
                    if loss != float('inf'):
                        print(f"  {name}: {loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"  -> New best loss: {best_loss:.4f}")
        
        # Reset model periodically
        if (step + 1) % 3 == 0:
            model.reset()
        
        # Save checkpoint
        if (step + 1) % save_every == 0:
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                f"{checkpoint_prefix}_step{step + 1}.pth"
            )
            torch.save(model.pipe.model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final checkpoint
    final_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_prefix}_final.pth")
    torch.save(model.pipe.model.state_dict(), final_path)
    print(f"Final checkpoint saved: {final_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Parallel Training for QwenAgentPlayer")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA adapters")
    parser.add_argument("--num_batches", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per framework")
    parser.add_argument("--num_parallel", type=int, default=4, help="Frameworks to run in parallel")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--save_every", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--print_every", type=int, default=100, help="Print progress every N steps")
    parser.add_argument("--checkpoint_prefix", type=str, default="qwen_agent_parallel", 
                        help="Checkpoint filename prefix")
    parser.add_argument("--load_checkpoint", type=str, default=None, 
                        help="Path to checkpoint to load")
    
    args = parser.parse_args()
    
    # Create model
    print("Creating model...")
    model = create_model(device=device, use_lora=False)
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        print(f"Loading checkpoint: {args.load_checkpoint}")
        model.pipe.model.load_state_dict(torch.load(args.load_checkpoint, map_location=device))
    
    # Get default frameworks
    frameworks = get_default_frameworks()
    
    # Train
    train_parallel(
        model=model,
        frameworks=frameworks,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        num_parallel=args.num_parallel,
        lr=args.lr,
        use_lora=args.use_lora,
        checkpoint_prefix=args.checkpoint_prefix,
        save_every=args.save_every,
        print_every=args.print_every,
    )


if __name__ == "__main__":
    main()
