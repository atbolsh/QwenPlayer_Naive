# Direction Names framework
# Task: Learn to associate action tokens with their names

from .general_framework import *
from .general_qa import *

prompts_for_action_names = [
    "Please go forward.<forward>",
    "Go forward:<forward>",
    "Please make the forward move<forward>",
    "Please progress<forward>",
    "Please turn clockwise <clock>",
    "Could you turn clockwise?<clock>Sure!",
    "Just take the CW move.<clock>",
    "Take the CW move.<clock>",
    "Please take the CW move.<clock>",
    "Please turn counter-clockwise <anticlock>",
    "Could you turn counter-clockwise?<anticlock>Sure!",
    "Just take the CCW move.<anticlock>",
    "Take the CCW move.<anticlock>",
    "Please take the CCW move.<anticlock>",
    "What action is <forward>? That's a move forward",
    "What action is <clock>? That's a CW turn",
    "What action is <clock>? That's a clockwise turn",
    "What action is <anticlock>? That's a CCW turn",
    "What action is <anticlock>? That's a counter-clockwise turn",
    "<forward> What action did you just take? Forward!",
    "<clock> What action did you just take? Clockwise turn!",
    "<anticlock> What action did you just take? Counterclockwise turn!",
    "<forward> What action did you just take? Forward move",
    "<clock> What action did you just take? Clockwise turn",
    "<anticlock> What action did you just take? I turned counter-clockwise, sir",
    "<forward> What action did you just take? Forward move",
    "<clock> What action did you just take? I turned clockwise, sir",
    "<anticlock> What action did you just take? Counter-clockwise turn",
    "<forward> What was that?? Forward move.",
    "<clock> What was that?? Clockwise turn",
    "<anticlock> What was that?? Counter-clockwise turn."
]

prompts_for_action_names_tensor = tensorify_list(prompts_for_action_names)


def _direction_names_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.pipe.model.eval()

    if training:
        model.pipe.model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")
    
    # Split batch across 2 generators: control + 1 task
    n_generators = 2
    chunk_size = batch_size // n_generators
    if chunk_size < 1:
        chunk_size = 1
    
    # Get settings and images for task chunk
    S_task = get_settings_batch(chunk_size)
    imgs_task = get_images(S_task)
    
    # Generate direction names texts
    texts_direction_names = simple_sample(chunk_size, prompts_for_action_names_tensor, device=device)
    
    # Get control texts and images
    ind = (batch_num * chunk_size) % num_controls
    if ind + chunk_size > num_controls:
        ind = num_controls - chunk_size
    control_texts = get_text_batch(sdt, ind, chunk_size)
    S_control = get_settings_batch(chunk_size)
    imgs_control = get_images(S_control)
    
    # Pad all texts to the same length before concatenation
    # Order: task, control (task first so demo images show task output)
    text_list = [texts_direction_names, control_texts]
    max_len = max(t.size(1) for t in text_list)
    padded_texts = []
    for t in text_list:
        if t.size(1) < max_len:
            pad = torch.zeros(t.size(0), max_len - t.size(1), dtype=t.dtype, device=t.device)
            t = torch.cat([t, pad], dim=1)
        padded_texts.append(t)
    
    # Concatenate all texts and images in consistent order
    # Order: task, control (task first so demo images show task output)
    all_texts = torch.cat(padded_texts, dim=0)
    all_imgs = torch.cat([imgs_task, imgs_control], dim=0)
    
    # Single forward pass with image reconstruction
    all_probs, all_recon = model_forward_with_tokens(model, all_texts, all_imgs, ret_imgs=True)
    
    # Compute text losses for each chunk
    # all_probs has shape (batch, vocab, seq_len) - slice on batch dimension (dim 0)
    text_losses = []
    for i in range(n_generators):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk_probs = all_probs[start_idx:end_idx, :, :]
        chunk_texts = all_texts[start_idx:end_idx]
        text_losses.append(get_text_loss(chunk_probs, chunk_texts))
    
    # Compute image loss
    img_loss = img_criterion(all_recon, all_imgs)
    
    # Total text loss
    text_loss = sum(text_losses)
    
    # Combined loss (same weighting as control framework)
    loss = img_loss + (text_loss / 1000)

    if training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.soft_reset()

    if printing:
        print(f"Total loss: {loss.item()} (img: {img_loss.item()}, text: {text_loss.item()}):\n"
              f"  {text_losses[0].item()} direction naming,\n"
              f"  {text_losses[1].item()} control\n")

    if reset_model:
        model.reset()

    return (loss.item(), text_losses[0].item(), text_losses[1].item(), img_loss.item())


def direction_names_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if compute_grad:
        return _direction_names_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _direction_names_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
