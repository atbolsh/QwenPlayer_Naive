#!/usr/bin/env python3
"""Quick script to test task1_img_sample and save input/output images."""

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import os

# Import the task1_img_sample function
from frameworks.arrow_to_gold import task1_img_sample

def save_tensor_as_image(tensor, filepath, scale=4):
    """Save a single image tensor to file.
    
    Args:
        tensor: Shape (3, H, W) in bfloat16, values 0-1
        filepath: Where to save
        scale: Upscale factor (nearest neighbor)
    """
    # Add batch dimension, upscale with nearest neighbor, then save
    img = tensor.unsqueeze(0).float()
    if scale > 1:
        img = F.interpolate(img, scale_factor=scale, mode='nearest')
    save_image(img[0], filepath)
    print(f"Saved: {filepath}")

def main():
    import math
    from game import discreteGame, BIG_tool_use_advanced_2_5
    
    # Create output directory if needed
    os.makedirs("demo_images", exist_ok=True)
    
    # Generate samples manually so we can also measure distances
    print("Generating arrow task samples with distance measurements...")
    G = discreteGame(BIG_tool_use_advanced_2_5)
    
    num_sample = 10
    distances = []
    
    for i in range(num_sample):
        bare_settings = G.random_bare_settings(gameSize=224, max_agent_offset=2.0)
        
        # Calculate distance (normalized 0-1 coords)
        ax, ay = bare_settings.agent_x, bare_settings.agent_y
        gx, gy = bare_settings.gold[0]
        dist = math.sqrt((ax - gx)**2 + (ay - gy)**2)
        distances.append(dist)
        
        G2 = discreteGame(bare_settings)
        img_in = torch.tensor(G2.getData(), dtype=torch.bfloat16)
        img_in = torch.permute(img_in, (2, 0, 1))  # HWC -> CHW
        
        G2.bare_draw_arrow_at_gold()
        img_out = torch.tensor(G2.getData(), dtype=torch.bfloat16)
        img_out = torch.permute(img_out, (2, 0, 1))  # HWC -> CHW
        
        save_tensor_as_image(img_in, f"demo_images/arrow_input_{i}.png")
        save_tensor_as_image(img_out, f"demo_images/arrow_output_{i}.png")
        print(f"  Sample {i}: agent=({ax:.3f}, {ay:.3f}), gold=({gx:.3f}, {gy:.3f}), distance={dist:.3f}")
    
    print(f"\nDistance stats:")
    print(f"  Min: {min(distances):.3f}")
    print(f"  Max: {max(distances):.3f}")
    print(f"  Mean: {sum(distances)/len(distances):.3f}")
    print(f"  Median: {sorted(distances)[len(distances)//2]:.3f}")
    print(f"  Count > 0.5: {sum(1 for d in distances if d > 0.5)}/{num_sample}")
    
    print(f"\nSaved {num_sample} input/output pairs to demo_images/")

if __name__ == "__main__":
    main()
