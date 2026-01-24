# Blue Line QA framework
# Task: Answer whether you are facing the blue line direction

from .general_framework import *
from .general_qa import *
from .game_logic_solver import true_angle_difference_magnitude
import math

prompts_blueLineDirection = [
    "Are you facing the blue line?",
    "Does your direction line up with the blue line?",
    "Are you facing where it's pointing?",
    "Are you facing in the right direction?"
]

Yreplies_blueLineDirection = ["Yep", "Absolutely.", "Certainly", "I think so.", "Uh-huh.", "Sure"]
Nreplies_blueLineDirection = ["Nuh-uh", "No", "I don't think so.", "Certainly not", "Absolutely not", "Nah"]

prompts_blueLineDirection_tensor = tensorify_list(prompts_blueLineDirection)
Yreplies_blueLineDirection_tensor = tensorify_list(Yreplies_blueLineDirection)
Nreplies_blueLineDirection_tensor = tensorify_list(Nreplies_blueLineDirection)

prompts_blueLineDirection_lens = get_lens(prompts_blueLineDirection_tensor)


def get_arrow_near_agent_direction(agent_direction):
    return G.mod2pi((np.random.random() * math.pi / 3) + agent_direction - (math.pi / 6))


def get_arrow_far_agent_direction(agent_direction):
    return G.mod2pi(agent_direction + math.pi / 6 + ((5 * math.pi / 3) * np.random.random()))


def get_random_directions(settings_batch):
    batchsize = len(settings_batch)
    deciders = (np.random.random((batchsize,)) < 0.5)
    directions = []
    for i in range(batchsize):
        if deciders[i]:
            directions.append(get_arrow_near_agent_direction(settings_batch[i].direction))
        else:
            directions.append(get_arrow_far_agent_direction(settings_batch[i].direction))
    return directions


def get_blue_line_direction_data(batch_size):
    S = get_settings_batch(batch_size) 
    directions = get_random_directions(S)
    deciderDict = {}
    for i in range(batch_size):
        theta = true_angle_difference_magnitude(directions[i], S[i].direction)
        same_direction = (theta < math.pi / 6)
        deciderDict[S[i]] = same_direction

    deciderFunc = lambda s: deciderDict[s]

    texts = text_generator_simple(
        S, prompts_blueLineDirection_tensor, Yreplies_blueLineDirection_tensor,
        Nreplies_blueLineDirection_tensor, prompts_blueLineDirection_lens, deciderFunc, device
    )

    imgs = torch.zeros(batch_size, 224, 224, 3)
    for i in range(batch_size):
        G2 = discreteGame(S[i])
        G2.draw_arrow(extension=1.0 + 3.0 * np.random.random(), direction=directions[i])
        imgs[i] = torch.tensor(G2.getData())
    imgs = torch.permute(imgs, (0, 3, 1, 2)).contiguous().to(device)
    return imgs, texts


def _blue_line_direction_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.pipe.model.eval()

    if training:
        model.pipe.model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")

    imgs, task_texts = get_blue_line_direction_data(batch_size)
        
    ind = (batch_num * batch_size) % num_controls
    if ind + batch_size > num_controls:
        ind = num_controls - batch_size
    control_texts = sdt[ind:ind + batch_size].to(device)

    all_texts = [control_texts, task_texts]
    text_inds = list(range(2))
    
    if random_order:
        random.shuffle(text_inds)

    all_probs = [0 for _ in text_inds]
    for t_ind in text_inds:
        all_probs[t_ind] = model_forward_with_tokens(model, all_texts[t_ind], imgs, ret_imgs=False)

    all_losses = [get_text_loss(all_probs[i], all_texts[i]) for i in range(2)]
    loss = sum(all_losses)

    if training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.soft_reset()

    if printing:
        print(f"Total loss: {loss.item()}:\n{all_losses[0].item()} control,\n{all_losses[1].item()} recognizing the blue line direction\n\n")

    if reset_model:
        model.reset()

    return (loss.item(), all_losses[0].item(), all_losses[1].item())


def blue_line_direction_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if compute_grad:
        return _blue_line_direction_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _blue_line_direction_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
