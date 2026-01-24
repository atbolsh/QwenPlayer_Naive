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

    imgs = torch.zeros(batch_size, 224, 224, 3)
    for i in range(batch_size):
        G2 = discreteGame(S[i])
        G2.draw_arrow(extension=1.0 + 3.0 * np.random.random(), direction=arrow_directions[i])
        imgs[i] = torch.tensor(G2.getData())
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

    imgs, ptbl_texts, ptg_texts, ptbla_texts, ptga_texts = get_please_turn_data(batch_size)
        
    ind = (batch_num * batch_size) % num_controls
    if ind + batch_size > num_controls:
        ind = num_controls - batch_size
    control_texts = sdt[ind:ind + batch_size].to(device)

    all_texts = [control_texts, ptbl_texts, ptg_texts, ptbla_texts, ptga_texts]
    text_inds = list(range(len(all_texts)))
    
    if random_order:
        random.shuffle(text_inds)

    all_probs = [0 for _ in text_inds]
    for t_ind in text_inds:
        all_probs[t_ind] = model_forward_with_tokens(model, all_texts[t_ind], imgs, ret_imgs=False)

    all_losses = [get_text_loss(all_probs[i], all_texts[i]) for i in range(len(all_texts))]
    loss = sum(all_losses)

    if training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.soft_reset()

    if printing:
        print(f"Total loss: {loss.item()}:\n{all_losses[0].item()} control,\n{all_losses[1].item()}" +
              f" turning towards blue line, {all_losses[2].item()} turning towards gold, " +
              f"{all_losses[3].item()} turning away from blue line " +
              f"{all_losses[4].item()} turning away from the gold.\n\n")

    if reset_model:
        model.reset()

    return (loss.item(), all_losses[0].item(), all_losses[1].item(), all_losses[2].item(), all_losses[3].item(), all_losses[4].item())


def please_turn_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if compute_grad:
        return _please_turn_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _please_turn_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
