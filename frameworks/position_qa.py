# Position QA framework (formerly tutorialQA_framework)
# Task: Answer questions about relative positions (left/right/up/down) of agent and gold

from .general_framework import *
from .general_qa import *

############
# Prompts and responses for position QA tasks

# Left-right gold position
task2_prompts_lrgold = [
    "Is the gold to the left or to the right of you?",
    "Which side is it on?",
    "Is it to the left or right of the agent?",
    "Do you need to go left or right to get the gold?",
    "Please tell me whether the gold is left or right.",
    "Please tell me which side is the gold on.",
    "Which side to you need to go to get it?",
    "Which side has gold?",
    "On which side is the gold?"
]

task2_Lreplies_lrgold = ["Left"]
task2_Rreplies_lrgold = ["Right"]

# Up-down gold position
task2_prompts_udgold = [
    "Is the gold above or below you?",
    "Is it up or down from the agent?",
    "Do you need to go up or down to get the gold?",
    "Please tell me whether the gold is above or below you.",
    "Please tell me whether the gold is up or down.",
    "Do you need to go up or down to get it?",
    "Which side has gold?",
    "On which side is the gold?"
]

task2_Ureplies_udgold = ["Up"]
task2_Dreplies_udgold = ["Down"]

# Left-right agent position
task2_prompts_lragent = [
    "Are you to the left or right of the gold?",
    "Which side is the gold on?",
    "Is the agent to the left or right of the gold?",
    "Please tell me whether you are right or left of the gold.",
    "Please tell me which side you are relative to the gold.",
    "On which side of the gold are you?"
]

task2_Lreplies_lragent = ["Left"]
task2_Rreplies_lragent = ["Right"]

# Up-down agent position
task2_prompts_udagent = [
    "Are you below or above the gold?",
    "Is the agent above or below the gold?",
    "Please tell me whether you are up or down from the gold.",
    "Please tell me whether you are above or below the gold."
]

task2_Ureplies_udagent = ["Up"]
task2_Dreplies_udagent = ["Down"]

########
# Tensorify prompts and responses

task2_prompts_lrgold_tensor = tensorify_list(task2_prompts_lrgold)
task2_Lreplies_lrgold_tensor = tensorify_list(task2_Lreplies_lrgold)
task2_Rreplies_lrgold_tensor = tensorify_list(task2_Rreplies_lrgold)

task2_prompts_udgold_tensor = tensorify_list(task2_prompts_udgold)
task2_Ureplies_udgold_tensor = tensorify_list(task2_Ureplies_udgold)
task2_Dreplies_udgold_tensor = tensorify_list(task2_Dreplies_udgold)

task2_prompts_lragent_tensor = tensorify_list(task2_prompts_lragent)
task2_Lreplies_lragent_tensor = tensorify_list(task2_Lreplies_lragent)
task2_Rreplies_lragent_tensor = tensorify_list(task2_Rreplies_lragent)

task2_prompts_udagent_tensor = tensorify_list(task2_prompts_udagent)
task2_Ureplies_udagent_tensor = tensorify_list(task2_Ureplies_udagent)
task2_Dreplies_udagent_tensor = tensorify_list(task2_Dreplies_udagent)

########
# Compute prompt lengths

task2_prompts_lrgold_lens = get_lens(task2_prompts_lrgold_tensor)
task2_prompts_udgold_lens = get_lens(task2_prompts_udgold_tensor)
task2_prompts_lragent_lens = get_lens(task2_prompts_lragent_tensor)
task2_prompts_udagent_lens = get_lens(task2_prompts_udagent_tensor)

########
# Decision functions

# Unintuitive, but pygame flips these
# This is 'left' and 'right' relative to the game setup, not the agent
is_gold_left = (lambda settings: settings.agent_y > settings.gold[0][1])
is_gold_up = (lambda settings: settings.agent_x > settings.gold[0][0])
is_agent_left = (lambda settings: not is_gold_left(settings))
is_agent_up = (lambda settings: not is_gold_up(settings))

########
# DPO text generators

task2_lrgold_generator_dpo = lambda settings_batch: text_generator_dpo(
    settings_batch, task2_prompts_lrgold_tensor, task2_Lreplies_lrgold_tensor,
    task2_Rreplies_lrgold_tensor, task2_prompts_lrgold_lens, is_gold_left, device)

task2_udgold_generator_dpo = lambda settings_batch: text_generator_dpo(
    settings_batch, task2_prompts_udgold_tensor, task2_Ureplies_udgold_tensor,
    task2_Dreplies_udgold_tensor, task2_prompts_udgold_lens, is_gold_up, device)

task2_lragent_generator_dpo = lambda settings_batch: text_generator_dpo(
    settings_batch, task2_prompts_lragent_tensor, task2_Lreplies_lragent_tensor,
    task2_Rreplies_lragent_tensor, task2_prompts_lragent_lens, is_agent_left, device)

task2_udagent_generator_dpo = lambda settings_batch: text_generator_dpo(
    settings_batch, task2_prompts_udagent_tensor, task2_Ureplies_udagent_tensor,
    task2_Dreplies_udagent_tensor, task2_prompts_udagent_lens, is_agent_up, device)

########

def _pad_to_len(tensor, target_len):
    if tensor.size(1) < target_len:
        pad = torch.zeros(tensor.size(0), target_len - tensor.size(1), dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad], dim=1)
    return tensor


def _qa_task_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.pipe.model.eval()

    if training:
        model.pipe.model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")
    
    # Split batch across 5 generators: 4 QA tasks + control
    n_generators = 5
    chunk_size = batch_size // n_generators
    if chunk_size < 1:
        chunk_size = 1
    
    # Get settings for each QA task (4 chunks)
    S_lrg = get_settings_batch(chunk_size)
    S_udg = get_settings_batch(chunk_size)
    S_lra = get_settings_batch(chunk_size)
    S_uda = get_settings_batch(chunk_size)
    
    # Get images for each chunk
    imgs_lrg = get_images(S_lrg)
    imgs_udg = get_images(S_udg)
    imgs_lra = get_images(S_lra)
    imgs_uda = get_images(S_uda)
    
    # Generate DPO texts for each chunk
    correct_lrg, wrong_lrg, lens_lrg = task2_lrgold_generator_dpo(S_lrg)
    correct_udg, wrong_udg, lens_udg = task2_udgold_generator_dpo(S_udg)
    correct_lra, wrong_lra, lens_lra = task2_lragent_generator_dpo(S_lra)
    correct_uda, wrong_uda, lens_uda = task2_udagent_generator_dpo(S_uda)
    
    # Get control texts and images
    ind = (batch_num * chunk_size) % num_controls
    if ind + chunk_size > num_controls:
        ind = num_controls - chunk_size
    control_texts = get_text_batch(sdt, ind, chunk_size)
    S_control = get_settings_batch(chunk_size)
    imgs_control = get_images(S_control)
    
    # Pad all texts to the same length
    correct_list = [correct_lrg, correct_udg, correct_lra, correct_uda]
    wrong_list = [wrong_lrg, wrong_udg, wrong_lra, wrong_uda]
    all_text_list = correct_list + wrong_list + [control_texts]
    max_len = max(t.size(1) for t in all_text_list)

    correct_list = [_pad_to_len(t, max_len) for t in correct_list]
    wrong_list = [_pad_to_len(t, max_len) for t in wrong_list]
    control_texts = _pad_to_len(control_texts, max_len)

    all_correct = torch.cat(correct_list, dim=0)  # (4*chunk_size, seq_len)
    all_wrong = torch.cat(wrong_list, dim=0)       # (4*chunk_size, seq_len)
    all_lens = torch.cat([lens_lrg, lens_udg, lens_lra, lens_uda], dim=0)

    all_texts = torch.cat([all_correct, control_texts], dim=0)
    all_imgs = torch.cat([imgs_lrg, imgs_udg, imgs_lra, imgs_uda, imgs_control], dim=0)
    
    # Single forward pass with image reconstruction
    all_probs, all_recon = model_forward_with_tokens(model, all_texts, all_imgs, ret_imgs=True)
    
    # DPO losses for each of the 4 task chunks
    task_total = 4 * chunk_size
    task_probs = all_probs[:task_total, :, :]
    task_correct = all_texts[:task_total]
    task_dpo_losses = []
    for i in range(4):
        s = i * chunk_size
        e = (i + 1) * chunk_size
        task_dpo_losses.append(get_dpo_text_loss(
            task_probs[s:e, :, :], task_correct[s:e], all_wrong[s:e], all_lens[s:e]
        ))

    # CE loss for control chunk
    control_probs = all_probs[task_total:, :, :]
    ctrl_texts = all_texts[task_total:]
    control_loss = get_text_loss(control_probs, ctrl_texts)
    
    img_loss = img_criterion(all_recon, all_imgs)
    task_dpo_total = sum(task_dpo_losses)
    text_loss = task_dpo_total + control_loss
    loss = img_loss + (text_loss / 1000)

    if training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.soft_reset()

    if printing:
        print(f"Total loss: {loss.item()} (img: {img_loss.item()}, text: {text_loss.item()}):\n"
              f"  {task_dpo_losses[0].item()} lrg (DPO),\n"
              f"  {task_dpo_losses[1].item()} udg (DPO),\n"
              f"  {task_dpo_losses[2].item()} lra (DPO),\n"
              f"  {task_dpo_losses[3].item()} uda (DPO),\n"
              f"  {control_loss.item()} control\n")

    if reset_model:
        model.reset()

    return (loss.item(), task_dpo_losses[0].item(), task_dpo_losses[1].item(), task_dpo_losses[2].item(), task_dpo_losses[3].item(), control_loss.item(), img_loss.item())


def qa_task_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if compute_grad:
        return _qa_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _qa_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
