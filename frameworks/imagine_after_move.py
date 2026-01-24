# Imagine After Move framework
# Task: Imagine what the scene will look like after a sequence of moves

from .general_framework import *
from .general_qa import *

prompts_imagineAfterMove = [
    "What will this game look if you go forward twice?",
    "Imagine the scene if you just step forward",
    "How would this look if you turn CW for a bit?",
    "How would this look if you turn CCW for a bit?",
    "Imagine this if you make a big turn clockwise",
    "How would this look after several CCW turns?",
    "Imagine this if you make a big turn counter-clockwise",
    "How would this look after several CW turns?",
    "How would this look if you turn CW one step?",
    "How would this look if you turn CCW one step?",
    "Imagine this if you make one step clockwise",
    "How would this look after one step CCW?",
    "Imagine this if you make one step counter-clockwise",
    "How would this look after one CW step?",
    "Woah, what would this look like if you step forward and then make a big turn CW?",
    "How does this look after 3 forward steps?",
    "Woah, imagine if you're looking the other way?",
    "Picture stepping forward for me.",
    "How do you figure this would look after turning CW just a bit?",
    "Imagine going forward then turning around.",
    "Imagine turning around then going forward",
    "What does this look like if you take 2 steps forward then one step CCW?"
]

prompts_imagineAfterMove_tensor = tensorify_list(prompts_imagineAfterMove)

CW_small = "<clock>" * 5
CCW_small = "<anticlock>" * 5
CW_big = CW_small * 2
CCW_big = CCW_small * 2
turn_around = CW_big * 3

move_instructions_imagineAfterMove = [
    "<forward><forward>",
    "<forward>",
    CW_small,
    CCW_small,
    CW_big,
    CCW_big,
    CCW_big,
    CW_big,
    "<clock>",
    "<anticlock>",
    "<clock>",
    "<anticlock>",
    "<anticlock>",
    "<clock>",
    "<forward>" + CW_big,
    "<forward><forward><forward>",
    turn_around,
    "<forward>",
    CW_small,
    "<forward>" + turn_around,
    turn_around + "<forward>",
    "<forward><forward><clock>"
]

# Tokenize on CPU - only used by game objects
move_instructions_imagineAfterMove_tensor = encode_batch(move_instructions_imagineAfterMove).to('cpu')


def process_steps(game, instructions):
    """Execute a sequence of action tokens on the game."""
    # Get token IDs for actions
    forward_id = tokenizer.convert_tokens_to_ids('<forward>')
    clock_id = tokenizer.convert_tokens_to_ids('<clock>')
    anticlock_id = tokenizer.convert_tokens_to_ids('<anticlock>')
    
    action_map = {forward_id: 1, clock_id: 3, anticlock_id: 4}
    
    for token_id in instructions.numpy():
        if token_id in action_map:
            game.actions[action_map[token_id]]()


def imagineAfterMove_data(batch_size):
    S = get_settings_batch(batch_size)
    imgs_in = get_images(S)

    num_prompts = prompts_imagineAfterMove_tensor.size()[0]
    inds = torch.randint(0, num_prompts, (batch_size,))
    texts = prompts_imagineAfterMove_tensor[inds]
    
    instructions = move_instructions_imagineAfterMove_tensor[inds]
    imgs_out = torch.zeros(batch_size, 224, 224, 3)
    for i in range(batch_size):
        G2 = discreteGame(S[i])
        process_steps(G2, instructions[i])
        imgs_out[i] = torch.tensor(G2.getData())
    imgs_out = torch.permute(imgs_out, (0, 3, 1, 2)).contiguous().to(device)

    return texts, imgs_in, imgs_out


def _imagineAfterMove_task_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.pipe.model.eval()

    if training:
        model.pipe.model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")

    task_texts, inp, out = imagineAfterMove_data(batch_size)
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


def imagineAfterMove_task_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if compute_grad:
        return _imagineAfterMove_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _imagineAfterMove_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
