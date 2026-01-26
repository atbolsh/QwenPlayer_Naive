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
    img_in = torch.zeros(num_sample, 224, 224, 3)
    img_out = torch.zeros(num_sample, 224, 224, 3)
    for i in range(num_sample):
        bare_settings = G.random_bare_settings(gameSize=224, max_agent_offset=0.5)
        G2 = discreteGame(bare_settings)
        img_in[i] = torch.tensor(G2.getData())
        G2.bare_draw_arrow_at_gold()
        img_out[i] = torch.tensor(G2.getData())
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

    inp, out, task_texts = task1_img_sample(batch_size)
    ind = (batch_num * batch_size) % num_controls
    if ind + batch_size > num_controls:
        ind = num_controls - batch_size
    control_texts = get_text_batch(sdt, ind, batch_size)

    # Added to fix a small bug
    ct_length = control_texts.size()[1]
    new_length = np.random.randint(6, ct_length)
    control_texts = control_texts[:, :new_length]

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


def arrow_task_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if compute_grad:
        return _arrow_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _arrow_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
