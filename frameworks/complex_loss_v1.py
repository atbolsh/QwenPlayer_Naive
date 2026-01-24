# Complex Loss V1 framework
# Task: Move agent closer to gold with differentiable image analysis

from .general_framework import *
from copy import deepcopy

# Import image analysis utilities
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from image_to_settings import get_agent_info, get_SINGLE_gold_info

prompts_move_agent_closer = [
    "Can you imagine if the agent is closer to the gold?",
    "Imagine the agent closer to the gold.",
    "Would this be easier if the agent were closer to the gold? Imagine it.",
    "Please imagine the agent somewhere closer to the gold."
]

complex_loss_text_tensor = encode_batch(prompts_move_agent_closer).contiguous().to(device)


def complex_loss_text_sample(num_sample=40):
    num_texts = complex_loss_text_tensor.size()[0]
    text_inds = torch.randint(0, num_texts, size=(num_sample,), device=device)
    texts = complex_loss_text_tensor[text_inds]
    return texts


def gamify_output(inp_settings_batch, agent_centers, directions, agent_radii, gold_centers, gold_radii, agent_filters, gold_filters):
    """Create game images from detected agent/gold positions."""
    N = len(inp_settings_batch)
    with torch.no_grad():
        out_settings_batch = []
        for i in range(N):
            setting = deepcopy(inp_settings_batch[i])

            if agent_filters[i]:
                setting.direction = directions[i].item()
                center = agent_centers[i].cpu().detach().numpy()
                setting.agent_x = center[0]
                setting.agent_y = center[1]
                setting.agent_r = agent_radii[i].item()

            if gold_filters[i]:
                gold_center = gold_centers[i].cpu().detach().numpy()
                setting.gold_centers = [[gold_center[0], gold_center[1]]]
                setting.gold_r = gold_radii[i].item()

            out_settings_batch.append(setting)

    return get_images(out_settings_batch)


def complex_loss_func(inp_settings_batch, agent_centers, directions, agent_radii, gold_centers, gold_radii, filters):
    """Compute complex loss penalizing various deviations."""
    N = len(inp_settings_batch)
    loss = torch.tensor(0.0).to(device)
    for i in range(N):
        if filters[i]:
            s = inp_settings_batch[i]
            G2 = discreteGame(s)
            loss += 100.0 * (agent_radii[i] - s.agent_r) ** 2
            loss += 100.0 * (gold_radii[i] - s.gold_r) ** 2
    
            gold_x, gold_y = s.gold[0]
            loss += 10.0 * ((gold_centers[i, 0] - gold_x) ** 2 + (gold_centers[i, 1] - gold_y) ** 2)
    
            old_distance_squared = (gold_x - s.agent_x) ** 2 + (gold_y - s.agent_y) ** 2
            new_distance_squared = (gold_x - agent_centers[i, 0]) ** 2 + (gold_y - agent_centers[i, 1]) ** 2
            loss += torch.relu((4 * new_distance_squared) - old_distance_squared)

    return loss


def _complex_loss_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.pipe.model.eval()

    if training:
        model.pipe.model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")
        
    inp_S = get_settings_batch(batch_size)
    inp_imgs = get_images(inp_S)

    texts = complex_loss_text_sample(batch_size)

    task_probs, task_recon = model_forward_with_tokens(model, texts, inp_imgs, ret_imgs=True)

    text_loss = get_text_loss(task_probs, texts)

    agent_centers, directions, agent_radii, agent_filters = get_agent_info(task_recon) 
    gold_centers, gold_radii, gold_filters = get_SINGLE_gold_info(task_recon, return_radii=True)

    filters = torch.logical_and(agent_filters, gold_filters)

    target_imgs = gamify_output(inp_S, agent_centers, directions, agent_radii, gold_centers, gold_radii, agent_filters, gold_filters)

    gameiness_loss = img_criterion(target_imgs, task_recon)

    CL = complex_loss_func(inp_S, agent_centers, directions, agent_radii, gold_centers, gold_radii, filters) 

    loss = (CL / 500) + gameiness_loss + (text_loss / 5000)

    if training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.soft_reset()
            
    if printing:
        print(f"Total loss: {loss.item()}; that's {CL.item()} task and {gameiness_loss.item()} 'gameiness' and {text_loss.item()} total text\n\n")

    if reset_model:
        model.reset()

    return loss.item(), CL.item(), gameiness_loss.item(), text_loss.item()


def complex_loss_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if compute_grad:
        return _complex_loss_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _complex_loss_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
