# Imagine Facing Gold framework
# Task: Imagine what the scene would look like if you faced the gold

from .general_framework import *
from .general_qa import *
from .game_logic_solver import gold_direction_angle

prompts_imagineFacingGold = [
    "Imagine facing the gold.",
    "What would this look like if you faced the gold.",
    "Damn, just picture yourself looking at the gold.",
    "How would this look if you faced the right way, eh?",
    "Can you imagine how this would look if you faced the gold?",
    "Imagine what it would look like if you were facing the gold"
]

prompts_imagineFacingGold_tensor = tensorify_list(prompts_imagineFacingGold)

########

def imagineFacingGold_data(batch_size):
    S = get_settings_batch(batch_size)
    imgs_in = get_images(S)
    imgs_out = torch.zeros(batch_size, 224, 224, 3)
    for i in range(batch_size):
        G2 = discreteGame(S[i])
        new_theta = gold_direction_angle(G2)
        S[i].direction = new_theta
        G2 = discreteGame(S[i])
        imgs_out[i] = torch.tensor(G2.getData())
    imgs_out = torch.permute(imgs_out, (0, 3, 1, 2)).contiguous().to(device)

    texts = simple_sample(batch_size, prompts_imagineFacingGold_tensor)
    return texts, imgs_in, imgs_out


def _imagineFacingGold_task_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.pipe.model.eval()

    if training:
        model.pipe.model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")

    task_texts, inp, out = imagineFacingGold_data(batch_size)
    ind = (batch_num * batch_size) % num_controls
    if ind + batch_size > num_controls:
        ind = num_controls - batch_size
    control_texts = sdt[ind:ind + batch_size].to(device)

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


def imagineFacingGold_task_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if compute_grad:
        return _imagineFacingGold_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _imagineFacingGold_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
