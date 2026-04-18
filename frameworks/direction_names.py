# Direction Names framework
# Task: Learn to associate action tokens with their names
#
# Two sub-tasks, each a 3-way classification (forward / clockwise / counter-clockwise):
#   A) Instruction: given a command, produce the correct action token
#   B) Naming: given an action token in context, produce the action's name

from .general_framework import *
from .general_qa import *

# --- Sub-task A: Instruction -> action token ---

prompts_instruction_forward = [
    "Go forward.",
    "Please go forward.",
]
prompts_instruction_cw = [
    "Turn clockwise.",
    "Please turn clockwise.",
]
prompts_instruction_ccw = [
    "Turn counter-clockwise.",
    "Please turn counter-clockwise.",
]

reply_forward_token = ["<forward>"]
reply_cw_token = ["<clock>"]
reply_ccw_token = ["<anticlock>"]

# Combine instruction prompts across all 3 actions into one prompt tensor per action
prompts_instruction_forward_tensor = tensorify_list(prompts_instruction_forward)
prompts_instruction_cw_tensor = tensorify_list(prompts_instruction_cw)
prompts_instruction_ccw_tensor = tensorify_list(prompts_instruction_ccw)

prompts_instruction_forward_lens = get_lens(prompts_instruction_forward_tensor)
prompts_instruction_cw_lens = get_lens(prompts_instruction_cw_tensor)
prompts_instruction_ccw_lens = get_lens(prompts_instruction_ccw_tensor)

reply_forward_token_tensor = tensorify_list(reply_forward_token)
reply_cw_token_tensor = tensorify_list(reply_cw_token)
reply_ccw_token_tensor = tensorify_list(reply_ccw_token)

# --- Sub-task B: Naming (action token in context -> name) ---

prompts_naming_forward = [
    "What action is <forward>?",
    "<forward> What action did you just take?",
    "<forward> What was that??",
]
prompts_naming_cw = [
    "What action is <clock>?",
    "<clock> What action did you just take?",
    "<clock> What was that??",
]
prompts_naming_ccw = [
    "What action is <anticlock>?",
    "<anticlock> What action did you just take?",
    "<anticlock> What was that??",
]

reply_forward_name = ["Forward"]
reply_cw_name = ["Clockwise"]
reply_ccw_name = ["Counter-clockwise"]

prompts_naming_forward_tensor = tensorify_list(prompts_naming_forward)
prompts_naming_cw_tensor = tensorify_list(prompts_naming_cw)
prompts_naming_ccw_tensor = tensorify_list(prompts_naming_ccw)

prompts_naming_forward_lens = get_lens(prompts_naming_forward_tensor)
prompts_naming_cw_lens = get_lens(prompts_naming_cw_tensor)
prompts_naming_ccw_lens = get_lens(prompts_naming_ccw_tensor)

reply_forward_name_tensor = tensorify_list(reply_forward_name)
reply_cw_name_tensor = tensorify_list(reply_cw_name)
reply_ccw_name_tensor = tensorify_list(reply_ccw_name)


########
# Data generation helpers

# Each sample randomly picks one of the 6 sub-tasks (3 instruction + 3 naming),
# then picks a random prompt from that sub-task's prompt pool.

# Organized as parallel lists so index 0=forward, 1=cw, 2=ccw
_instruction_prompts = [prompts_instruction_forward_tensor, prompts_instruction_cw_tensor, prompts_instruction_ccw_tensor]
_instruction_lens = [prompts_instruction_forward_lens, prompts_instruction_cw_lens, prompts_instruction_ccw_lens]
_instruction_replies = [reply_forward_token_tensor, reply_cw_token_tensor, reply_ccw_token_tensor]

_naming_prompts = [prompts_naming_forward_tensor, prompts_naming_cw_tensor, prompts_naming_ccw_tensor]
_naming_lens = [prompts_naming_forward_lens, prompts_naming_cw_lens, prompts_naming_ccw_lens]
_naming_replies = [reply_forward_name_tensor, reply_cw_name_tensor, reply_ccw_name_tensor]


def _generate_direction_names_texts(batch_size):
    """Generate prompt+reply texts for direction naming tasks.

    Each sample randomly chooses one of 6 sub-tasks (instruction or naming,
    for each of the 3 actions). Returns (correct_texts, wrong_texts, prompt_lens).
    """
    all_prompt_pools = _instruction_prompts + _naming_prompts  # 6 pools
    all_lens_pools = _instruction_lens + _naming_lens
    all_reply_pools = _instruction_replies + _naming_replies

    max_prompt_size = max(p.size(1) for p in all_prompt_pools)
    max_reply_size = max(r.size(1) for r in all_reply_pools)
    total_len = max_prompt_size + max_reply_size

    correct_tensor = torch.zeros((batch_size, total_len), device=device, dtype=all_prompt_pools[0].dtype)
    wrong_tensor = torch.zeros((batch_size, total_len), device=device, dtype=all_prompt_pools[0].dtype)
    lens_tensor = torch.zeros(batch_size, dtype=torch.long, device=device)

    for i in range(batch_size):
        subtask = random.randint(0, 5)
        group = subtask // 3  # 0 = instruction, 1 = naming
        action_idx = subtask % 3  # 0 = forward, 1 = cw, 2 = ccw

        prompts = all_prompt_pools[subtask]
        prompt_lengths = all_lens_pools[subtask]
        correct_reply = all_reply_pools[subtask][0]

        wrong_action_idx = random.choice([j for j in range(3) if j != action_idx])
        wrong_reply_pool_idx = group * 3 + wrong_action_idx
        wrong_reply = all_reply_pools[wrong_reply_pool_idx][0]

        ind = torch.randint(0, prompts.size(0), (1,)).item()
        prompt = prompts[ind]
        length = prompt_lengths[ind]
        lens_tensor[i] = length

        _stitch(prompt, correct_reply, correct_tensor[i], length)
        _stitch(prompt, wrong_reply, wrong_tensor[i], length)

    return correct_tensor.contiguous(), wrong_tensor.contiguous(), lens_tensor


def _direction_names_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.pipe.model.eval()

    if training:
        model.pipe.model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")
    
    # Split batch across 2 generators: task + control; remainder goes to a random chunk
    n_generators = 2
    chunk_size = batch_size // n_generators
    if chunk_size < 1:
        chunk_size = 1
    remainder = batch_size - n_generators * chunk_size
    chunk_sizes = [chunk_size] * n_generators
    if remainder > 0:
        chunk_sizes[random.randint(0, n_generators - 1)] += remainder
    
    # Task chunk
    S_task = get_settings_batch(chunk_sizes[0])
    imgs_task = get_images(S_task)
    correct_texts, wrong_texts, prompt_lens = _generate_direction_names_texts(chunk_sizes[0])
    
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
              f"  {task_dpo_loss.item()} direction naming (DPO),\n"
              f"  {control_loss.item()} control\n")

    if reset_model:
        model.reset()

    return (loss.item(), task_dpo_loss.item(), control_loss.item(), img_loss.item())


def direction_names_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if compute_grad:
        return _direction_names_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _direction_names_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
