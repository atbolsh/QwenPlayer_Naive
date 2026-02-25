# Please Turn QA framework
# Task: Turn towards or away from gold/blue line

from .general_framework import *
from .general_qa import *
from .game_logic_solver import should_turn_anticlockwise_forward, should_turn_anticlockwise_forward_ENGINE
from copy import deepcopy
import math

prompts_pleaseTurnGold = ["Please turn towards the gold", "Turn towards the gold", "Face towards gold"]
prompts_pleaseTurnBlueLine = ["Please turn towards the blue line", "Turn towards the blue line", "Face towards the blue line"]
prompts_pleaseTurnGoldAWAY = ["Please turn away from the gold", "Turn away from the gold", "Face away from gold"]
prompts_pleaseTurnBlueLineAWAY = ["Please turn away from the blue line", "Turn away from the blue line", "Face away from the blue line"]

replies_clockwise = ["<clock>"]
replies_counterclockwise = ["<anticlock>"]

########

prompts_pleaseTurnGold_tensor = tensorify_list(prompts_pleaseTurnGold)
prompts_pleaseTurnGold_lens = get_lens(prompts_pleaseTurnGold_tensor)

prompts_pleaseTurnBlueLine_tensor = tensorify_list(prompts_pleaseTurnBlueLine)
prompts_pleaseTurnBlueLine_lens = get_lens(prompts_pleaseTurnBlueLine_tensor)

prompts_pleaseTurnGoldAWAY_tensor = tensorify_list(prompts_pleaseTurnGoldAWAY)
prompts_pleaseTurnGoldAWAY_lens = get_lens(prompts_pleaseTurnGoldAWAY_tensor)

prompts_pleaseTurnBlueLineAWAY_tensor = tensorify_list(prompts_pleaseTurnBlueLineAWAY)
prompts_pleaseTurnBlueLineAWAY_lens = get_lens(prompts_pleaseTurnBlueLineAWAY_tensor)

replies_clockwise_tensor = tensorify_list(replies_clockwise)
replies_counterclockwise_tensor = tensorify_list(replies_counterclockwise)

########

best_turn_cw = lambda settings: not should_turn_anticlockwise_forward(discreteGame(deepcopy(settings)))

pleaseTurnGold_generator_simple = lambda settings_batch: text_generator_simple(
    settings_batch, prompts_pleaseTurnGold_tensor, replies_clockwise_tensor,
    replies_counterclockwise_tensor, prompts_pleaseTurnGold_lens, best_turn_cw, device
)

pleaseTurnGoldAWAY_generator_simple = lambda settings_batch: text_generator_simple(
    settings_batch, prompts_pleaseTurnGoldAWAY_tensor, replies_counterclockwise_tensor,
    replies_clockwise_tensor, prompts_pleaseTurnGoldAWAY_lens, best_turn_cw, device
)

########

def get_please_turn_data(batch_size):
    S = get_settings_batch(batch_size) 
    arrow_directions = 2 * math.pi * np.random.random((batch_size,))
    deciderDict_CWarrow = {}
    for i in range(batch_size):
        deciderDict_CWarrow[S[i]] = not should_turn_anticlockwise_forward_ENGINE(S[i].direction, arrow_directions[i])

    deciderFunc_CWarrow = lambda s: deciderDict_CWarrow[s]

    texts_pleaseTurnBlueLine = text_generator_simple(
        S, prompts_pleaseTurnBlueLine_tensor, replies_clockwise_tensor,
        replies_counterclockwise_tensor, prompts_pleaseTurnBlueLine_lens, deciderFunc_CWarrow, device
    )
    texts_pleaseTurnGold = pleaseTurnGold_generator_simple(S)
    texts_pleaseTurnBlueLineAWAY = text_generator_simple(
        S, prompts_pleaseTurnBlueLineAWAY_tensor, replies_counterclockwise_tensor,
        replies_clockwise_tensor, prompts_pleaseTurnBlueLineAWAY_lens, deciderFunc_CWarrow, device
    )
    texts_pleaseTurnGoldAWAY = pleaseTurnGoldAWAY_generator_simple(S)

    imgs = torch.zeros(batch_size, 224, 224, 3, dtype=torch.float32)
    for i in range(batch_size):
        G2 = discreteGame(S[i])
        G2.draw_arrow(extension=1.0 + 3.0 * np.random.random(), direction=arrow_directions[i])
        imgs[i] = torch.tensor(G2.getData(), dtype=torch.float32)
    imgs = torch.permute(imgs, (0, 3, 1, 2)).contiguous().to(device)
    return imgs, texts_pleaseTurnBlueLine, texts_pleaseTurnGold, texts_pleaseTurnBlueLineAWAY, texts_pleaseTurnGoldAWAY


def _please_turn_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.pipe.model.eval()

    if training:
        model.pipe.model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")

    # 5 generators: 4 turn tasks (shared images) + control
    # Turn tasks share images, so they must all be the same size; remainder goes to control
    n_generators = 5
    chunk_size = batch_size // n_generators
    if chunk_size < 1:
        chunk_size = 1
    ctrl_size = batch_size - 4 * chunk_size
    chunk_sizes = [chunk_size] * 4 + [ctrl_size]

    # Generate turn task data (shared settings/images for all 4 turn tasks)
    imgs_ptbl, ptbl_texts, ptg_texts, ptbla_texts, ptga_texts = get_please_turn_data(chunk_size)

    # Control chunk (absorbs remainder)
    ind = (batch_num * ctrl_size) % num_controls
    if ind + ctrl_size > num_controls:
        ind = num_controls - ctrl_size
    control_texts = get_text_batch(sdt, ind, ctrl_size)
    S_control = get_settings_batch(ctrl_size)
    imgs_control = get_images(S_control)

    # Order: ptbl, ptg, ptbla, ptga, control
    text_list = [ptbl_texts, ptg_texts, ptbla_texts, ptga_texts, control_texts]
    img_list = [imgs_ptbl, imgs_ptbl, imgs_ptbl, imgs_ptbl, imgs_control]

    # Pad texts to same length
    max_len = max(t.size(1) for t in text_list)
    padded_texts = []
    for t in text_list:
        if t.size(1) < max_len:
            pad = torch.zeros(t.size(0), max_len - t.size(1), dtype=t.dtype, device=t.device)
            t = torch.cat([t, pad], dim=1)
        padded_texts.append(t)

    all_texts = torch.cat(padded_texts, dim=0)
    all_imgs = torch.cat(img_list, dim=0)

    # Single forward pass
    all_probs, all_recon = model_forward_with_tokens(model, all_texts, all_imgs, ret_imgs=True)

    # Compute text losses per chunk
    text_losses = []
    offset = 0
    for cs in chunk_sizes:
        chunk_probs = all_probs[offset:offset + cs, :, :]
        chunk_texts = all_texts[offset:offset + cs]
        text_losses.append(get_text_loss(chunk_probs, chunk_texts))
        offset += cs

    # Image loss
    img_loss = img_criterion(all_recon, all_imgs)
    text_loss = sum(text_losses)
    loss = img_loss + (text_loss / 1000)

    if training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.soft_reset()

    if printing:
        print(f"Total loss: {loss.item()} (img: {img_loss.item()}, text: {text_loss.item()}):\n"
              f"  {text_losses[0].item()} turning towards blue line,\n"
              f"  {text_losses[1].item()} turning towards gold,\n"
              f"  {text_losses[2].item()} turning away from blue line,\n"
              f"  {text_losses[3].item()} turning away from gold,\n"
              f"  {text_losses[4].item()} control\n")

    if reset_model:
        model.reset()

    return (loss.item(), text_losses[4].item(), text_losses[0].item(), text_losses[1].item(), text_losses[2].item(), text_losses[3].item())


def please_turn_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if compute_grad:
        return _please_turn_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _please_turn_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
