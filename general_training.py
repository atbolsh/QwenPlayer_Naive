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
import torch.nn.functional as F
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
DEFAULT_INIT_CHECKPOINT = "brain_checkpoints/qwen_agent_scales_control_only_batch2000_merged.pth"
#DEFAULT_INIT_CHECKPOINT = "brain_checkpoints/qwen_agent_control_arrow_qa_batch10000.pth"

# ============================================================
# EASILY EDITABLE: Save prefix for checkpoints and CSV
# ============================================================
DEFAULT_SAVE_PREFIX = "qwen_agent_blue_line_v3_from_control"

# Directories
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "brain_checkpoints")
DEMO_DIR = os.path.join(os.path.dirname(__file__), "demo_images")


def merge_lora_checkpoint(lora_state_dict: dict, lora_alpha: int = 16, r: int = 4) -> dict:
    """
    Convert a LoRA-wrapped checkpoint to a standard (non-LoRA) checkpoint
    using PEFT's official merge_and_unload() method.
    
    This creates a temporary model, loads the LoRA checkpoint, calls PEFT's
    merge_and_unload() to properly merge LoRA weights, then extracts the
    merged state dict.
    
    Args:
        lora_state_dict: State dict saved from a model with LoRA applied
        lora_alpha: LoRA alpha scaling factor (default: 16, matching apply_lora_to_text)
        r: LoRA rank (default: 4, matching apply_lora_to_text)
        
    Returns:
        Merged state dict compatible with non-LoRA models
    """
    # Create a fresh model with LoRA to use PEFT's merge functionality
    temp_model = create_model(use_lora=True)
    
    # Load the LoRA state dict into the model
    temp_model.pipe.model.load_state_dict(lora_state_dict, strict=False)
    
    # Use PEFT's merge_and_unload on the qwen_model (the part with LoRA)
    # This properly merges LoRA weights into the base weights
    merged_qwen = temp_model.pipe.model.qwen_model.merge_and_unload()
    
    # Rebuild the full state dict with merged qwen weights
    merged = {}
    
    # Add qwen model weights (now merged, with standard key names)
    for key, value in merged_qwen.state_dict().items():
        merged[f'qwen_model.{key}'] = value
    
    # Add non-LoRA components from original state dict
    for key, value in lora_state_dict.items():
        if key.startswith('img_enc') or key.startswith('img_dec') or key == 'layer_scale_factors':
            merged[key] = value
    
    # Clean up
    del temp_model
    torch.cuda.empty_cache()
    
    return merged
LEDGER_PATH = os.path.join(os.path.dirname(__file__), f"{DEFAULT_SAVE_PREFIX}_losses.csv")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DEMO_DIR, exist_ok=True)


def save_demo_images(model, step: int, task_name: str, prompt: str = "What do you see?"):
    """
    Save sample input/output images for a given task.
    
    Images are saved at 4x resolution (896x896) using nearest-neighbor upscaling
    to preserve sharp pixel boundaries without blurring.
    
    Args:
        model: QwenAgentPlayer instance
        step: Current training step (for filename)
        task_name: Name of the task (for filename)
        prompt: Text prompt to use
    """
    model.pipe.model.eval()
    with torch.no_grad():
        # Generate a game image (bf16)
        settings = G.random_bare_settings(gameSize=224, max_agent_offset=2.0)
        img = get_images([settings], device=device)
        
        # Run model forward
        tokens = encode_batch([prompt]).to(device)
        _, img_recon = model_forward_with_tokens(model, tokens, img, ret_imgs=True)
        
        # Save images (convert to float32 for torchvision)
        safe_task_name = task_name.replace("_batch", "").replace("_task", "")
        input_path = os.path.join(DEMO_DIR, f"step_{step:06d}_{safe_task_name}_input.png")
        output_path = os.path.join(DEMO_DIR, f"step_{step:06d}_{safe_task_name}_output.png")
        
        # Scale up 4x using nearest-neighbor (no blurring/interpolation)
        # This preserves sharp pixel boundaries for easier defect inspection
        img_scaled = F.interpolate(img.float(), scale_factor=4, mode='nearest')
        save_image(img_scaled[0], input_path)
        
        if img_recon is not None:
            recon_scaled = F.interpolate(img_recon.float().clamp(0, 1), scale_factor=4, mode='nearest')
            save_image(recon_scaled[0], output_path)
        
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
    # NOTE: LoRA is now applied in main() BEFORE checkpoint loading
    # This is necessary because checkpoints saved with LoRA have different key structure
    # The use_lora parameter is kept for API compatibility but not used here
    
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
            
            print("=" * 60)
            
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
            state_dict = model.pipe.model.state_dict()
            
            # Save LoRA checkpoint (or standard if not using LoRA)
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, 
                f"{checkpoint_prefix}_batch{b + 1}.pth"
            )
            torch.save(state_dict, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            
            # If using LoRA, also save a merged version
            if use_lora:
                merged_state = merge_lora_checkpoint(state_dict)
                merged_path = os.path.join(
                    CHECKPOINT_DIR,
                    f"{checkpoint_prefix}_merged_batch{b + 1}.pth"
                )
                torch.save(merged_state, merged_path)
                print(f"Merged checkpoint saved: {merged_path}")
            
            # Reset canvases before generating demo images to avoid issues
            model.reset()
            
            # Save demo images for ALL active frameworks
            print("Saving demo images for active frameworks...")
            for func, _ in frameworks:
                if func in FRAMEWORK_DEMO_INFO:
                    demo_name, demo_prompt = FRAMEWORK_DEMO_INFO[func]
                    try:
                        input_path, output_path = save_demo_images(model, b + 1, demo_name, demo_prompt)
                        print(f"  Demo images saved: {demo_name}")
                    except Exception as e:
                        print(f"  Error saving demo images for {demo_name}: {e}")
            
            # Reset canvases after demo images
            model.reset()
    
    # Save final checkpoint
    state_dict = model.pipe.model.state_dict()
    final_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_prefix}_final.pth")
    torch.save(state_dict, final_path)
    print(f"Final checkpoint saved: {final_path}")
    
    # If using LoRA, also save a merged version
    if use_lora:
        merged_state = merge_lora_checkpoint(state_dict)
        merged_final_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_prefix}_merged_final.pth")
        torch.save(merged_state, merged_final_path)
        print(f"Merged final checkpoint saved: {merged_final_path}")
    
    # Reset canvases before generating final demo images
    model.reset()
    
    # Save final demo images for ALL active frameworks
    print("Saving final demo images for active frameworks...")
    for func, _ in frameworks:
        if func in FRAMEWORK_DEMO_INFO:
            demo_name, demo_prompt = FRAMEWORK_DEMO_INFO[func]
            try:
                save_demo_images(model, num_batches, f"final_{demo_name}", demo_prompt)
                print(f"  Final demo saved: {demo_name}")
            except Exception as e:
                print(f"  Error saving final demo images for {demo_name}: {e}")
    
    # Reset canvases after demo images
    model.reset()
    
    return model


# ============================================================
# FRAMEWORK DEMO INFO: Maps framework functions to (name, prompt) for demo images
# Names match the framework file names (without .py) for easy identification
# ============================================================
FRAMEWORK_DEMO_INFO = {
    control_batch: ("control", "What do you see?"),
    arrow_task_batch: ("arrow_to_gold", "Draw the path to the gold."),
    qa_task_batch: ("position_qa", "What do you see?"),
    mem_canvas_batch: ("mem_canvas_use", "What do you see?"),
    blue_line_direction_batch: ("blue_line_qa", "Which direction is the blue line?"),
    gold_direction_batch: ("gold_direction_qa", "Which direction is the gold?"),
    gold_proximity_batch: ("near_gold_qa", "How close is the gold?"),
    please_turn_batch: ("please_turn_qa", "Please turn."),
    relposition_qa_batch: ("relposition_qa", "Where are you?"),
    direction_names_batch: ("direction_names", "Name the directions."),
    zoom_task_batch: ("zoom", "Zoom in."),
    imagineWithoutYou_task_batch: ("imagine_without_you", "Show me the room without yourself."),
    imagineWithoutGold_task_batch: ("imagine_without_gold", "Show me the room without the gold."),
    imagineWithoutWalls_task_batch: ("imagine_without_walls", "Show me the room without walls."),
    imagineWallsOnly_task_batch: ("imagine_walls_only", "Show me just the walls."),
    imagineFacingGold_task_batch: ("imagine_facing_gold", "Show me facing the gold."),
    imagineCloser2Gold_task_batch: ("imagine_closer_to_gold", "Show me closer to the gold."),
    imagineAfterMove_task_batch: ("imagine_after_move", "Show me after moving."),
}


def get_default_frameworks() -> List[Tuple[Callable, int]]:
    """
    Get default framework configuration.
    
    Comment/uncomment frameworks here - demo images will automatically
    be generated for all active frameworks using FRAMEWORK_DEMO_INFO above.
    """
    return [
        (control_batch, 8),
        (arrow_task_batch, 24),
        (qa_task_batch, 8),
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
    parser.add_argument("--batch_size", type=int, default=60, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--save_every", type=int, default=1000, help="Save checkpoint every N batches")
    parser.add_argument("--print_every", type=int, default=100, help="Print progress every N batches")
    parser.add_argument("--checkpoint_prefix", type=str, default=DEFAULT_SAVE_PREFIX, help="Checkpoint filename prefix")
    parser.add_argument("--load_checkpoint", type=str, default=DEFAULT_INIT_CHECKPOINT, 
                        help="Path to checkpoint to load (default: DEFAULT_INIT_CHECKPOINT at top of file)")
    
    args = parser.parse_args()
    
    # Helper to detect if checkpoint is LoRA format
    def is_lora_checkpoint(state_dict):
        for key in state_dict.keys():
            if 'lora_A' in key or 'lora_B' in key or 'base_layer' in key:
                return True
        return False
    
    # Load and inspect checkpoint first (if specified)
    checkpoint_state = None
    checkpoint_is_lora = False
    if args.load_checkpoint and args.load_checkpoint.strip():
        checkpoint_path = args.load_checkpoint
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
            checkpoint_is_lora = is_lora_checkpoint(checkpoint_state)
            print(f"  Checkpoint format: {'LoRA' if checkpoint_is_lora else 'non-LoRA'}")
        else:
            print(f"WARNING: Checkpoint not found: {checkpoint_path}")
            print("Using weights loaded at import time from frameworks/general_framework.py")
    
    # Handle all 4 cases:
    # 1) Non-LoRA checkpoint + --use_lora → Load, then apply LoRA
    # 2) LoRA checkpoint + --use_lora → Apply LoRA first, then load directly
    # 3) Non-LoRA checkpoint + no --use_lora → Load and train normally
    # 4) LoRA checkpoint + no --use_lora → Merge weights, warn, train without LoRA
    
    print("Creating model...")
    model = create_model(device=device, use_lora=False)
    
    if checkpoint_state is not None:
        if checkpoint_is_lora and args.use_lora:
            # Case 2: LoRA checkpoint + --use_lora → Apply LoRA first, then load
            print("Case 2: LoRA checkpoint with --use_lora")
            print("  Applying LoRA adapters first...")
            model = apply_lora_to_text(model)
            print("  Loading LoRA checkpoint...")
            load_result = model.pipe.model.load_state_dict(checkpoint_state, strict=False)
            
        elif checkpoint_is_lora and not args.use_lora:
            # Case 4: LoRA checkpoint + no --use_lora → Merge and train without LoRA
            print("=" * 60)
            print("NOTICE: LoRA checkpoint detected but --use_lora not set!")
            print("Merging LoRA weights into base model for non-LoRA training.")
            print("=" * 60)
            merged_state = merge_lora_checkpoint(checkpoint_state)
            print(f"  Merged {len(checkpoint_state)} LoRA keys → {len(merged_state)} standard keys")
            load_result = model.pipe.model.load_state_dict(merged_state, strict=False)
            
        elif not checkpoint_is_lora and args.use_lora:
            # Case 1: Non-LoRA checkpoint + --use_lora → Load first, then apply LoRA
            print("Case 1: Non-LoRA checkpoint with --use_lora")
            print("  Loading non-LoRA checkpoint first...")
            load_result = model.pipe.model.load_state_dict(checkpoint_state, strict=False)
            print("  Applying LoRA adapters on top of loaded weights...")
            model = apply_lora_to_text(model)
            
        else:
            # Case 3: Non-LoRA checkpoint + no --use_lora → Simple load
            print("Case 3: Non-LoRA checkpoint without --use_lora")
            load_result = model.pipe.model.load_state_dict(checkpoint_state, strict=False)
        
        # Diagnostic output
        if load_result.missing_keys:
            print(f"  WARNING: Missing keys (first 5): {load_result.missing_keys[:5]}...")
        if load_result.unexpected_keys:
            print(f"  INFO: Unexpected keys (first 5): {load_result.unexpected_keys[:5]}...")
        if not load_result.missing_keys and not load_result.unexpected_keys:
            print(f"  All keys matched successfully!")
    
    elif args.use_lora:
        # No checkpoint but --use_lora requested
        print("No checkpoint specified, applying LoRA to fresh model...")
        model = apply_lora_to_text(model)
    
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
