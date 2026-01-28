# Imagine Closer to Gold framework
# Task: Imagine being halfway to the gold

from .general_framework import *
from .general_qa import *
from .game_logic_solver import gold_direction_angle
from copy import deepcopy

prompts_imagineCloser2Gold = [
    "Damn, just picture being closer to the reward.",
    "What if you were half way there, though?",
    "Imagine being halfway there, huh.",
    "Imagine being half way to the gold.",
    "Picture yourself being much closer to the gold."
]

prompts_imagineCloser2Gold_tensor = tensorify_list(prompts_imagineCloser2Gold)

########

def get_new_setting_imagineCloser2Gold(s):
    s2 = deepcopy(s)
    s2.direction = gold_direction_angle(discreteGame(s))
    s2.agent_x = 0.5 * (s.agent_x + s.gold[0][0])
    s2.agent_y = 0.5 * (s.agent_y + s.gold[0][1])
    return s2


def imagineCloser2Gold_data(batch_size):
    S = get_settings_batch(batch_size)
    imgs_in = get_images(S)
    imgs_out = torch.zeros(batch_size, 224, 224, 3, dtype=torch.bfloat16)
    for i in range(batch_size):
        G2 = discreteGame(get_new_setting_imagineCloser2Gold(S[i]))
        imgs_out[i] = torch.tensor(G2.getData(), dtype=torch.bfloat16)
    imgs_out = torch.permute(imgs_out, (0, 3, 1, 2)).contiguous().to(device)

    texts = simple_sample(batch_size, prompts_imagineCloser2Gold_tensor)
    return texts, imgs_in, imgs_out


def _imagineCloser2Gold_task_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.pipe.model.eval()

    if training:
        model.pipe.model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")

    task_texts, inp, out = imagineCloser2Gold_data(batch_size)
    ind = (batch_num * batch_size) % num_controls
    if ind + batch_size > num_controls:
        ind = num_controls - batch_size
    control_texts = get_text_batch(sdt, ind, batch_size)

    flip = 0
    if random_order:
        flip += random.randint(0, 1)

    if flip:
        task_probs, task_recon = model_forward_with_tokens(model, task_texts, inp, ret_imgs=True)
        control_probs, control_recon = model_forward_with_tokens(model, control_texts, inp, ret_imgs=True)
    else:
        control_probs, control_recon = model_forward_with_tokens(model, control_texts, inp, ret_imgs=True)
        task_probs, task_recon = model_forward_with_tokens(model, task_texts, inp, ret_imgs=True)

    l1 = img_criterion(task_recon, out)
    l2 = img_criterion(control_recon, inp)
    img_loss = l1 + l2
    tl1 = get_text_loss(task_probs, task_texts)
    tl2 = get_text_loss(control_probs, control_texts)
    text_loss = tl1 + tl2
    loss = img_loss + (text_loss / 5000)

    if training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.soft_reset()
    
    if printing:
        print(f"Total loss: {loss.item()}; that's {l1.item()} task and {l2.item()} recon and {text_loss.item()} total text\n\n")

    if reset_model:
        model.reset()

    return loss.item(), l1.item(), l2.item(), tl1.item(), tl2.item()


def imagineCloser2Gold_task_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if compute_grad:
        return _imagineCloser2Gold_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _imagineCloser2Gold_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
