# Arrow to Gold framework (formerly tutorial1_framework)
# Task: Draw a line from the agent to the nearest gold

from .general_framework import *

task1_prompts = [
    "Imagine the line from the agent to the nearest gold.",
    "What's the straight path from the agent to the gold?",
    "Please draw the straight line to the gold from the agent.",
    "How would you move from the agent to the gold?",
    "What's a direct path from the agent to the gold?",
    "From the agent to the nearest coin, please draw a path."
]

task1_text_tensor = encode_batch(task1_prompts).contiguous().to(device)

########

def task1_img_sample(num_sample=40):
    """Generate input/output image pairs for arrow-to-gold task."""
    img_in = torch.zeros(num_sample, 224, 224, 3, dtype=torch.bfloat16)
    img_out = torch.zeros(num_sample, 224, 224, 3, dtype=torch.bfloat16)
    for i in range(num_sample):
        bare_settings = G.random_bare_settings(gameSize=224, max_agent_offset=0.9)
        G2 = discreteGame(bare_settings)
        img_in[i] = torch.tensor(G2.getData(), dtype=torch.bfloat16)
        G2.bare_draw_arrow_at_gold()
        img_out[i] = torch.tensor(G2.getData(), dtype=torch.bfloat16)
    img_in = torch.permute(img_in, (0, 3, 1, 2)).contiguous().to(device)
    img_out = torch.permute(img_out, (0, 3, 1, 2)).contiguous().to(device)
    num_texts = task1_text_tensor.size()[0]
    text_inds = torch.randint(0, num_texts, size=(num_sample,), device=device)
    texts = task1_text_tensor[text_inds]
    # Randomly pad or not
    if random.randint(0, 1):
        text_length = texts.size()[1]
        target = torch.zeros(num_sample, 32, dtype=texts.dtype, device=texts.device)
        target[:, :text_length] += texts
        return img_in, img_out, target
    else:
        return img_in, img_out, texts

########

def _arrow_task_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
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
    
    # Get task data: inp_task -> out_task (with arrow drawn)
    inp_task, out_task, task_texts = task1_img_sample(chunk_size)
    
    # Get control data: inp_control -> inp_control (reconstruction)
    ind = (batch_num * chunk_size) % num_controls
    if ind + chunk_size > num_controls:
        ind = num_controls - chunk_size
    control_texts = get_text_batch(sdt, ind, chunk_size)
    
    # Added to fix a small bug
    ct_length = control_texts.size()[1]
    new_length = np.random.randint(6, ct_length)
    control_texts = control_texts[:, :new_length]
    
    # Generate control images (just for reconstruction, target = input)
    S_control = get_settings_batch(chunk_size)
    inp_control = get_images(S_control)
    
    # Pad texts to same length for concatenation
    task_len = task_texts.size()[1]
    ctrl_len = control_texts.size()[1]
    max_len = max(task_len, ctrl_len)
    
    if task_len < max_len:
        pad = torch.zeros(chunk_size, max_len - task_len, dtype=task_texts.dtype, device=task_texts.device)
        task_texts = torch.cat([task_texts, pad], dim=1)
    if ctrl_len < max_len:
        pad = torch.zeros(chunk_size, max_len - ctrl_len, dtype=control_texts.dtype, device=control_texts.device)
        control_texts = torch.cat([control_texts, pad], dim=1)
    
    # Concatenate all inputs, targets, and texts in consistent order
    # Order: control, task
    all_inputs = torch.cat([inp_control, inp_task], dim=0)
    all_targets = torch.cat([inp_control, out_task], dim=0)  # control reconstructs input, task outputs arrow
    all_texts = torch.cat([control_texts, task_texts], dim=0)
    
    # Single forward pass with image reconstruction
    all_probs, all_recon = model_forward_with_tokens(model, all_texts, all_inputs, ret_imgs=True)
    
    # Compute image losses for each chunk
    control_recon = all_recon[:chunk_size]
    task_recon = all_recon[chunk_size:]
    l2 = img_criterion(control_recon, inp_control)  # control reconstruction
    l1 = img_criterion(task_recon, out_task)  # task arrow drawing
    img_loss = l1 + l2
    
    # Compute text losses for each chunk
    # all_probs has shape (batch, vocab, seq_len) - slice on batch dimension (dim 0)
    control_probs = all_probs[:chunk_size, :, :]
    task_probs = all_probs[chunk_size:, :, :]
    tl2 = get_text_loss(control_probs, control_texts)
    tl1 = get_text_loss(task_probs, task_texts)
    text_loss = tl1 + tl2
    
    loss = img_loss + (text_loss / 5000)

    if training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.soft_reset()
    
    if printing:
        print(f"Total loss: {loss.item()} (img: {img_loss.item()}, text: {text_loss.item()}):\n"
              f"  {l1.item()} task arrow,\n"
              f"  {l2.item()} control recon,\n"
              f"  {tl1.item()} task text,\n"
              f"  {tl2.item()} control text\n")

    if reset_model:
        model.reset()

    return loss.item(), l1.item(), l2.item(), tl1.item(), tl2.item(), img_loss.item()


def arrow_task_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if compute_grad:
        return _arrow_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _arrow_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
