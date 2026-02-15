#!/usr/bin/env python3
"""
Dedicated training script for VisionWeightedSum on the canvas recall task.

Freezes all parameters except img_weight, trains using mem_canvas_batch
with train_weights=True (CrossEntropy on canvas_weights).

Usage:
    python train_mem_canvas.py
    python train_mem_canvas.py --batch_size 30 --lr 1e-4 --num_batches 50000
    python train_mem_canvas.py --load_checkpoint brain_checkpoints/some_other.pth
"""

import os
import csv
import argparse
import warnings

import torch
import torch.optim as optim
from torchvision.utils import save_image
import torch.nn.functional as F

from frameworks import (
    device, create_model, tokenizer,
    model_forward_with_tokens, encode_batch,
    G, get_images,
    mem_canvas_batch, control_batch,
)

warnings.filterwarnings('ignore')

# ============================================================
# Defaults
# ============================================================
CHECKPOINT_DIR = "brain_checkpoints"
DEFAULT_INIT_CHECKPOINT = "brain_checkpoints/qwen_agent_vision_weights_initialized.pth"
DEFAULT_SAVE_PREFIX = "qwen_agent_mem_canvas_weights"
LEDGER_PATH = os.path.join(os.path.dirname(__file__), f"{DEFAULT_SAVE_PREFIX}_losses.csv")


def _convert_checkpoint_bf16_to_float32(state_dict):
    """Convert any bf16 tensors in a state dict to float32."""
    for k, v in state_dict.items():
        if v.dtype == torch.bfloat16:
            state_dict[k] = v.float()


def save_demo_images(model, batch_num, prefix="mem_canvas"):
    """Save a quick demo image to verify training progress."""
    demo_dir = os.path.join("demo_images", prefix)
    os.makedirs(demo_dir, exist_ok=True)

    settings = G.random_bare_settings(gameSize=224, max_agent_offset=2.0)
    from game import discreteGame
    G2 = discreteGame(settings)
    img = torch.tensor(G2.getData(), dtype=torch.float32).unsqueeze(0)
    img = torch.permute(img, (0, 3, 1, 2)).contiguous().to(device)

    prompt = "What do you see right now?"
    encoded = tokenizer([prompt], padding='max_length', truncation=True,
                        max_length=32, return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    attention_mask = (input_ids != pad_id).long()

    with torch.no_grad():
        result = model.batch_forward(
            input_ids=input_ids, image=img,
            attention_mask=attention_mask, generate_image=True,
        )

    gen = result.get('generated_images')
    if gen is not None:
        # Upscale 4x for visibility
        gen_up = F.interpolate(gen.float().clamp(0, 1), scale_factor=4, mode='nearest')
        img_up = F.interpolate(img.float().clamp(0, 1), scale_factor=4, mode='nearest')
        save_image(img_up, os.path.join(demo_dir, f"input_batch{batch_num}.png"))
        save_image(gen_up, os.path.join(demo_dir, f"output_batch{batch_num}.png"))


def main():
    parser = argparse.ArgumentParser(description="Train VisionWeightedSum on canvas recall")
    parser.add_argument("--num_batches", type=int, default=10000000, help="Number of training batches")
    parser.add_argument("--batch_size", type=int, default=120, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_every", type=int, default=1000, help="Save checkpoint every N batches")
    parser.add_argument("--print_every", type=int, default=100, help="Print progress every N batches")
    parser.add_argument("--checkpoint_prefix", type=str, default=DEFAULT_SAVE_PREFIX)
    parser.add_argument("--load_checkpoint", type=str, default=DEFAULT_INIT_CHECKPOINT)

    args = parser.parse_args()

    # ---- Load model and checkpoint ----
    print("Creating model...")
    model = create_model(device=device, use_lora=False)

    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        print(f"Loading checkpoint: {args.load_checkpoint}")
        ckpt = torch.load(args.load_checkpoint, weights_only=True, map_location=device)
        _convert_checkpoint_bf16_to_float32(ckpt)
        load_result = model.pipe.model.load_state_dict(ckpt, strict=False)
        if load_result.missing_keys:
            print(f"  Missing keys (first 5): {load_result.missing_keys[:5]}...")
        if load_result.unexpected_keys:
            print(f"  Unexpected keys (first 5): {load_result.unexpected_keys[:5]}...")
        if not load_result.missing_keys and not load_result.unexpected_keys:
            print("  All keys matched!")
    else:
        print(f"WARNING: Checkpoint not found: {args.load_checkpoint}")
        print("Using weights from frameworks import (frankenstein checkpoint).")

    # ---- Freeze everything, unfreeze only img_weight ----
    for param in model.pipe.model.parameters():
        param.requires_grad = False
    for param in model.pipe.model.img_weight.parameters():
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.pipe.model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.pipe.model.parameters())
    print(f"Trainable parameters: {n_trainable:,} / {n_total:,} total")

    # ---- Optimizer (only img_weight params) ----
    optimizer = optim.Adam(model.pipe.model.img_weight.parameters(), lr=args.lr, eps=1e-9)

    # ---- CSV ledger ----
    ledger_path = os.path.join(os.path.dirname(__file__),
                               f"{args.checkpoint_prefix}_losses.csv")
    with open(ledger_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['batch', 'total_loss', 'img_recall_loss', 'text_loss', 'weight_ce_loss'])

    # ---- Training loop ----
    print(f"\nStarting training for up to {args.num_batches} batches...")
    print(f"  batch_size={args.batch_size}, lr={args.lr}")
    print(f"  save_every={args.save_every}, print_every={args.print_every}")
    print(f"  Enduring checkpoints every {10 * args.save_every} batches")
    print("=" * 60)

    prev_checkpoint_path = None
    prev_checkpoint_batch = None

    total_loss_accum = 0.0
    loss_count = 0

    for b in range(args.num_batches):
        # Reset every 3rd batch so canvases can accumulate
        reset_model = (b % 3 == 2)

        should_print = ((b + 1) % args.print_every == 0) or (b < 10)

        try:
            results = mem_canvas_batch(
                batch_size=args.batch_size,
                model=model,
                optimizer=optimizer,
                batch_num=b,
                compute_grad=True,
                training=True,
                model_eval=False,
                reset_model=reset_model,
                printing=should_print,
                train_weights=True,
            )
            total_loss, img_loss, text_loss, weight_ce_loss = results
            total_loss_accum += total_loss
            loss_count += 1
        except Exception as e:
            print(f"Error at batch {b + 1}: {e}")
            model.reset()
            continue

        # Log to CSV every print_every
        if should_print:
            avg_loss = total_loss_accum / max(loss_count, 1)
            print(f"  Batch {b + 1}: avg_loss={avg_loss:.4f}")

            with open(ledger_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([b + 1, total_loss, img_loss, text_loss, weight_ce_loss])

        # Reset accumulators periodically
        if (b + 1) % args.print_every == 0:
            total_loss_accum = 0.0
            loss_count = 0

        # ---- Save checkpoint ----
        if (b + 1) % args.save_every == 0:
            # Unfreeze all before saving so the full state dict is complete
            for param in model.pipe.model.parameters():
                param.requires_grad = True

            state_dict = model.pipe.model.state_dict()
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                f"{args.checkpoint_prefix}_batch{b + 1}.pth"
            )
            torch.save(state_dict, checkpoint_path)
            is_enduring = (b + 1) % (10 * args.save_every) == 0
            print(f"Checkpoint saved: {checkpoint_path}" +
                  (" (enduring)" if is_enduring else ""))

            # Delete previous non-enduring checkpoint
            if (prev_checkpoint_path is not None and
                    prev_checkpoint_batch % (10 * args.save_every) != 0):
                if os.path.exists(prev_checkpoint_path):
                    os.remove(prev_checkpoint_path)
                    print(f"  Deleted previous: {prev_checkpoint_path}")

            prev_checkpoint_path = checkpoint_path
            prev_checkpoint_batch = b + 1

            # Save demo images
            model.reset()
            try:
                save_demo_images(model, b + 1, args.checkpoint_prefix)
                print("  Demo images saved.")
            except Exception as e:
                print(f"  Error saving demo images: {e}")
            model.reset()

            # Re-freeze for continued training
            for param in model.pipe.model.parameters():
                param.requires_grad = False
            for param in model.pipe.model.img_weight.parameters():
                param.requires_grad = True

    # ---- Save final checkpoint ----
    for param in model.pipe.model.parameters():
        param.requires_grad = True

    state_dict = model.pipe.model.state_dict()
    final_path = os.path.join(CHECKPOINT_DIR, f"{args.checkpoint_prefix}_final.pth")
    torch.save(state_dict, final_path)
    print(f"\nFinal checkpoint saved: {final_path}")

    model.reset()
    try:
        save_demo_images(model, args.num_batches, f"final_{args.checkpoint_prefix}")
        print("Final demo images saved.")
    except Exception as e:
        print(f"Error saving final demo images: {e}")

    print("Done!")


if __name__ == "__main__":
    main()
