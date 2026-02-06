#!/usr/bin/env python3
"""
Merge LoRA Checkpoint Script

Converts a LoRA-wrapped checkpoint to a standard (non-LoRA) checkpoint by
merging LoRA weights into base weights using the PEFT formula:
    W_merged = W_base + (lora_B @ lora_A) * (lora_alpha / r)

Usage:
    python merge_lora_checkpoint.py input_checkpoint.pth output_checkpoint.pth
    
    # Or with default output name (adds _merged suffix):
    python merge_lora_checkpoint.py input_checkpoint.pth
    
    # With custom LoRA parameters:
    python merge_lora_checkpoint.py input.pth output.pth --lora_alpha 32 --r 8
"""

import argparse
import os
import torch

# Import the merge function from general_training to avoid code duplication
from general_training import merge_lora_checkpoint


def is_lora_checkpoint(state_dict: dict) -> bool:
    """Check if a checkpoint contains LoRA weights."""
    for key in state_dict.keys():
        if 'lora_A' in key or 'lora_B' in key or 'base_layer' in key:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights into base weights to create a standard checkpoint"
    )
    parser.add_argument(
        "input_checkpoint",
        type=str,
        help="Path to the LoRA checkpoint to merge"
    )
    parser.add_argument(
        "output_checkpoint",
        type=str,
        nargs='?',
        default=None,
        help="Path for the output merged checkpoint (default: input_name_merged.pth)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling factor (default: 16)"
    )
    parser.add_argument(
        "--r",
        type=int,
        default=4,
        help="LoRA rank (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Generate default output name if not provided
    if args.output_checkpoint is None:
        base, ext = os.path.splitext(args.input_checkpoint)
        args.output_checkpoint = f"{base}_merged{ext}"
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.input_checkpoint}")
    state_dict = torch.load(args.input_checkpoint, map_location='cpu', weights_only=True)
    print(f"Loaded {len(state_dict)} keys")
    
    # Check if it's a LoRA checkpoint
    if not is_lora_checkpoint(state_dict):
        print("WARNING: This doesn't appear to be a LoRA checkpoint.")
        print("No LoRA keys (lora_A, lora_B, base_layer) found.")
        print("Checkpoint may already be in standard format.")
        return
    
    # Merge
    print("\nMerging LoRA weights...")
    scaling = args.lora_alpha / args.r
    print(f"Using scaling factor: {scaling} (lora_alpha={args.lora_alpha}, r={args.r})")
    merged = merge_lora_checkpoint(state_dict, lora_alpha=args.lora_alpha, r=args.r)
    print(f"Merged checkpoint has {len(merged)} keys")
    
    # Save
    print(f"\nSaving to: {args.output_checkpoint}")
    torch.save(merged, args.output_checkpoint)
    print("Done!")
    
    # Show some sample keys
    print("\nSample keys in merged checkpoint:")
    for key in sorted(merged.keys())[:10]:
        print(f"  {key}")


if __name__ == "__main__":
    main()
