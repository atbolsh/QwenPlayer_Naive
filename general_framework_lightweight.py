# Lightweight game utilities - no Qwen model loading
# Import this for image-only training without the heavy model overhead

import torch
import torch.nn as nn

from game import discreteGame, BIG_tool_use_advanced_2_5

########
# Device
########

device = torch.device('cuda:0')  # CHANGE THIS EVERY TIME
# device = torch.device('cuda:1')  # CHANGE THIS EVERY TIME

########
# Game setup
########

game_settings = BIG_tool_use_advanced_2_5
game_settings.gameSize = 224  # for compatibility with brain's expected size
G = discreteGame(game_settings)

########
# Loss functions (image only)
########

img_criterion = nn.MSELoss()

########
# Game utilities
########

def get_settings_batch(batch_size, bare=True, restrict_angles=True):
    if bare:
        return [G.random_bare_settings(gameSize=224, max_agent_offset=0.5) for i in range(batch_size)]
    else:
        return [G.random_settings(gameSize=224, restrict_angles=restrict_angles) for i in range(batch_size)]


def get_images(settings_batch=None, device=device, ignore_agent=False, ignore_gold=False, ignore_walls=False, batch_size=None, bare=True, restrict_angles=True):
    # If no settings provided, generate them using bare/restrict_angles flags
    if settings_batch is None:
        if batch_size is None:
            raise ValueError("Must provide either settings_batch or batch_size")
        settings_batch = get_settings_batch(batch_size, bare=bare, restrict_angles=restrict_angles)
    
    batch_size = len(settings_batch)
    img = torch.zeros(batch_size, 224, 224, 3)
    should_draw = (ignore_agent or ignore_gold or ignore_walls)
    for i in range(batch_size):
        G2 = discreteGame(settings_batch[i])
        if should_draw:
            G2.draw(ignore_agent=ignore_agent, ignore_gold=ignore_gold, ignore_wals=ignore_walls)
        img[i] = torch.tensor(G2.getData())
    img = torch.permute(img, (0, 3, 1, 2)).contiguous().to(device)
    return img
