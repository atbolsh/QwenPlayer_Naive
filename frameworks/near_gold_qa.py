# Near Gold QA framework
# Task: Answer whether you are near the gold

from .general_framework import *
from .general_qa import *

prompts_goldProximity = [
    "Are you near the gold?",
    "Does your direction line up with the gold?",
    "Are so close to the gold you're salivating?",
    "Is the meal right in front of you?",
    "Are you almost at the reward?",
    "The coin's right there, yes?",
]

Yreplies_goldProximity = ["Yes"]
Nreplies_goldProximity = ["No"]

prompts_goldProximity_tensor = tensorify_list(prompts_goldProximity)
Yreplies_goldProximity_tensor = tensorify_list(Yreplies_goldProximity)
Nreplies_goldProximity_tensor = tensorify_list(Nreplies_goldProximity)

prompts_goldProximity_lens = get_lens(prompts_goldProximity_tensor)


def gold_is_near(s):
    return (((s.agent_x - s.gold[0][0])**2 + (s.agent_y - s.gold[0][1])**2) < 0.15 * 0.15)


def get_gold_proximity_data(batch_size):
    S = get_settings_batch(batch_size)

    correct_texts, wrong_texts, prompt_lens = text_generator_dpo(
        S, prompts_goldProximity_tensor, Yreplies_goldProximity_tensor,
        Nreplies_goldProximity_tensor, prompts_goldProximity_lens, gold_is_near, device
    )
    imgs = get_images(S)

    return imgs, correct_texts, wrong_texts, prompt_lens


def _gold_proximity_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.pipe.model.eval()

    if training:
        model.pipe.model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")

    # 2 generators: task + control; remainder goes to a random chunk
    n_generators = 2
    chunk_size = batch_size // n_generators
    if chunk_size < 1:
        chunk_size = 1
    remainder = batch_size - n_generators * chunk_size
    chunk_sizes = [chunk_size] * n_generators
    if remainder > 0:
        chunk_sizes[random.randint(0, n_generators - 1)] += remainder

    # Task chunk
    imgs_task, correct_texts, wrong_texts, prompt_lens = get_gold_proximity_data(chunk_sizes[0])

    # Control chunk
    ind = (batch_num * chunk_sizes[1]) % num_controls
    if ind + chunk_sizes[1] > num_controls:
        ind = num_controls - chunk_sizes[1]
    control_texts = get_text_batch(sdt, ind, chunk_sizes[1])
    S_control = get_settings_batch(chunk_sizes[1])
    imgs_control = get_images(S_control)

    # Pad texts to same length
    text_list = [correct_texts, control_texts]
    max_len = max(t.size(1) for t in text_list)
    padded_texts = []
    padded_wrong = wrong_texts
    for t in text_list:
        if t.size(1) < max_len:
            pad = torch.zeros(t.size(0), max_len - t.size(1), dtype=t.dtype, device=t.device)
            t = torch.cat([t, pad], dim=1)
        padded_texts.append(t)
    if padded_wrong.size(1) < max_len:
        pad = torch.zeros(padded_wrong.size(0), max_len - padded_wrong.size(1), dtype=padded_wrong.dtype, device=padded_wrong.device)
        padded_wrong = torch.cat([padded_wrong, pad], dim=1)

    all_texts = torch.cat(padded_texts, dim=0)
    all_imgs = torch.cat([imgs_task, imgs_control], dim=0)

    # Single forward pass
    all_probs, all_recon = model_forward_with_tokens(model, all_texts, all_imgs, ret_imgs=True)

    # DPO loss for task chunk
    task_probs = all_probs[:chunk_sizes[0], :, :]
    task_texts = all_texts[:chunk_sizes[0]]
    task_dpo_loss = get_dpo_text_loss(task_probs, task_texts, padded_wrong, prompt_lens)

    # CE loss for control chunk
    control_probs = all_probs[chunk_sizes[0]:, :, :]
    ctrl_texts = all_texts[chunk_sizes[0]:]
    control_loss = get_text_loss(control_probs, ctrl_texts)

    img_loss = img_criterion(all_recon, all_imgs)
    text_loss = task_dpo_loss + control_loss
    loss = img_loss + (text_loss / 1000)

    if training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.soft_reset()

    if printing:
        print(f"Total loss: {loss.item()} (img: {img_loss.item()}, text: {text_loss.item()}):\n"
              f"  {task_dpo_loss.item()} gold proximity (DPO),\n"
              f"  {control_loss.item()} control\n")

    if reset_model:
        model.reset()

    return (loss.item(), control_loss.item(), task_dpo_loss.item())


def gold_proximity_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if compute_grad:
        return _gold_proximity_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _gold_proximity_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
