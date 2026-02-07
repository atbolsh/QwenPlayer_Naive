# Gold Direction QA framework
# Task: Answer whether you are facing the gold

from .general_framework import *
from .general_qa import *
from .game_logic_solver import will_intersect_forward

prompts_goldDirection = [
    "Are you facing the gold?",
    "Does your direction line up with the gold?",
    "Are you facing where the gold is?",
    "Are you facing in the right direction?"
]

Yreplies_goldDirection = ["Yep", "Absolutely.", "Certainly", "I think so.", "Uh-huh.", "Sure"]
Nreplies_goldDirection = ["Nuh-uh", "No", "I don't think so.", "Certainly not", "Absolutely not", "Nah"]

prompts_goldDirection_tensor = tensorify_list(prompts_goldDirection)
Yreplies_goldDirection_tensor = tensorify_list(Yreplies_goldDirection)
Nreplies_goldDirection_tensor = tensorify_list(Nreplies_goldDirection)

prompts_goldDirection_lens = get_lens(prompts_goldDirection_tensor)


def get_gold_direction_data(batch_size):
    S = get_settings_batch(batch_size) 
    deciderFunc = lambda s: will_intersect_forward(discreteGame(s))

    texts = text_generator_simple(
        S, prompts_goldDirection_tensor, Yreplies_goldDirection_tensor,
        Nreplies_goldDirection_tensor, prompts_goldDirection_lens, deciderFunc, device
    )
    imgs = get_images(S)

    return imgs, texts


def _gold_direction_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.pipe.model.eval()

    if training:
        model.pipe.model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")

    # Split batch across 2 generators: control + 1 QA task
    n_generators = 2
    chunk_size = batch_size // n_generators
    if chunk_size < 1:
        chunk_size = 1
    
    # Get task data (images and texts together)
    imgs_task, task_texts = get_gold_direction_data(chunk_size)
    
    # Get control texts and images
    ind = (batch_num * chunk_size) % num_controls
    if ind + chunk_size > num_controls:
        ind = num_controls - chunk_size
    control_texts = get_text_batch(sdt, ind, chunk_size)
    S_control = get_settings_batch(chunk_size)
    imgs_control = get_images(S_control)
    
    # Pad all texts to the same length before concatenation
    text_list = [control_texts, task_texts]
    max_len = max(t.size(1) for t in text_list)
    padded_texts = []
    for t in text_list:
        if t.size(1) < max_len:
            pad = torch.zeros(t.size(0), max_len - t.size(1), dtype=t.dtype, device=t.device)
            t = torch.cat([t, pad], dim=1)
        padded_texts.append(t)
    
    # Concatenate all texts and images in consistent order
    # Order: control, task
    all_texts = torch.cat(padded_texts, dim=0)
    all_imgs = torch.cat([imgs_control, imgs_task], dim=0)
    
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
              f"  {text_losses[0].item()} control,\n"
              f"  {text_losses[1].item()} recognizing gold direction\n")

    if reset_model:
        model.reset()

    return (loss.item(), text_losses[0].item(), text_losses[1].item(), img_loss.item())


def gold_direction_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if compute_grad:
        return _gold_direction_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _gold_direction_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
