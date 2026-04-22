# Relative Position QA framework
# Task: Answer questions about relative position and required moves

from .general_framework import *
from .general_qa import *
from .game_logic_solver import will_intersect_forward, should_turn_anticlockwise_forward, best_move_forward
from copy import deepcopy

prompts_willIntersectForward = [
    "If you go forward, will you eat?",
    "Is the gold in your path?",
    "How do you figure, will you intersect it just by going forward?",
    "Is going forward enough?",
    "Can you get the gold without turning?",
    "You don't need to turn, right?"
]

Yreplies_willIntersectForward = ["Yes"]
Nreplies_willIntersectForward = ["No"]

prompts_whichWayTurn = [
    "Which way should you turn, do you figure?",
    "Damn, how can I twist in the right direction?",
    "Which way to fix our direction?",
    "How should you turn?"
]

CWreplies_whichWayTurn = ["Clockwise"]
CCWreplies_whichWayTurn = ["Counter-clockwise"]

prompts_whatNextMove = [
    "Damn it, what's the move here, partner?",
    "What should you do here?",
    "In this position, what should you do?",
    "How do you figure, what's the move for us?",
    "What's the move?"
]

Freplies_whatNextMove = ["Forward"]
CWreplies_whatNextMove = CWreplies_whichWayTurn
CCWreplies_whatNextMove = CCWreplies_whichWayTurn

########

prompts_willIntersectForward_tensor = tensorify_list(prompts_willIntersectForward)
Yreplies_willIntersectForward_tensor = tensorify_list(Yreplies_willIntersectForward)
Nreplies_willIntersectForward_tensor = tensorify_list(Nreplies_willIntersectForward)

prompts_whichWayTurn_tensor = tensorify_list(prompts_whichWayTurn)
CWreplies_whichWayTurn_tensor = tensorify_list(CWreplies_whichWayTurn)
CCWreplies_whichWayTurn_tensor = tensorify_list(CCWreplies_whichWayTurn)

prompts_whatNextMove_tensor = tensorify_list(prompts_whatNextMove)
Freplies_whatNextMove_tensor = tensorify_list(Freplies_whatNextMove)
CWreplies_whatNextMove_tensor = tensorify_list(CWreplies_whatNextMove)
CCWreplies_whatNextMove_tensor = tensorify_list(CCWreplies_whatNextMove)

########

prompts_willIntersectForward_lens = get_lens(prompts_willIntersectForward_tensor)
prompts_whichWayTurn_lens = get_lens(prompts_whichWayTurn_tensor)
prompts_whatNextMove_lens = get_lens(prompts_whatNextMove_tensor)

########

willIntersectForward = lambda settings: will_intersect_forward(discreteGame(deepcopy(settings)))
best_turn_cw = lambda settings: not should_turn_anticlockwise_forward(discreteGame(deepcopy(settings)))

throwaway_index_helper = {1: 0, 3: 1, 4: 2}
best_move = lambda settings: throwaway_index_helper[best_move_forward(discreteGame(deepcopy(settings)))]

########
# DPO text generators

willIntersectForward_generator_dpo = lambda settings_batch: text_generator_dpo(
    settings_batch, prompts_willIntersectForward_tensor, Yreplies_willIntersectForward_tensor,
    Nreplies_willIntersectForward_tensor, prompts_willIntersectForward_lens, willIntersectForward, device
)

whichWayTurn_generator_dpo = lambda settings_batch: text_generator_dpo(
    settings_batch, prompts_whichWayTurn_tensor, CWreplies_whichWayTurn_tensor,
    CCWreplies_whichWayTurn_tensor, prompts_whichWayTurn_lens, best_turn_cw, device
)

whatNextMove_generator_dpo = lambda settings_batch: text_generator_dpo_GENERAL(
    settings_batch, prompts_whatNextMove_tensor,
    [Freplies_whatNextMove_tensor, CWreplies_whatNextMove_tensor, CCWreplies_whatNextMove_tensor],
    prompts_whatNextMove_lens, best_move, device
)

########

def _pad_to_len(tensor, target_len):
    if tensor.size(1) < target_len:
        pad = torch.zeros(tensor.size(0), target_len - tensor.size(1), dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad], dim=1)
    return tensor


def _relposition_qa_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.pipe.model.eval()

    if training:
        model.pipe.model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")
    
    # Split batch across 4 generators: 3 QA tasks + control; remainder goes to a random chunk
    n_generators = 4
    chunk_size = batch_size // n_generators
    if chunk_size < 1:
        chunk_size = 1
    remainder = batch_size - n_generators * chunk_size
    chunk_sizes = [chunk_size] * n_generators
    if remainder > 0:
        chunk_sizes[random.randint(0, n_generators - 1)] += remainder
    
    # Get settings for each QA task
    S_wif = get_settings_batch(chunk_sizes[0])
    S_wwt = get_settings_batch(chunk_sizes[1])
    S_wnm = get_settings_batch(chunk_sizes[2])
    
    # Get images for each chunk
    imgs_wif = get_images(S_wif)
    imgs_wwt = get_images(S_wwt)
    imgs_wnm = get_images(S_wnm)
    
    # Generate DPO texts for each chunk
    correct_wif, wrong_wif, lens_wif = willIntersectForward_generator_dpo(S_wif)
    correct_wwt, wrong_wwt, lens_wwt = whichWayTurn_generator_dpo(S_wwt)
    correct_wnm, wrong_wnm, lens_wnm = whatNextMove_generator_dpo(S_wnm)
    
    # Control chunk
    ind = (batch_num * chunk_sizes[3]) % num_controls
    if ind + chunk_sizes[3] > num_controls:
        ind = num_controls - chunk_sizes[3]
    control_texts = get_text_batch(sdt, ind, chunk_sizes[3])
    S_control = get_settings_batch(chunk_sizes[3])
    imgs_control = get_images(S_control)
    
    # Pad all texts to the same length
    correct_list = [correct_wif, correct_wwt, correct_wnm]
    wrong_list = [wrong_wif, wrong_wwt, wrong_wnm]
    all_text_list = correct_list + wrong_list + [control_texts]
    max_len = max(t.size(1) for t in all_text_list)

    correct_list = [_pad_to_len(t, max_len) for t in correct_list]
    wrong_list = [_pad_to_len(t, max_len) for t in wrong_list]
    control_texts = _pad_to_len(control_texts, max_len)

    all_correct = torch.cat(correct_list, dim=0)
    all_wrong = torch.cat(wrong_list, dim=0)
    all_lens = torch.cat([lens_wif, lens_wwt, lens_wnm], dim=0)

    all_texts = torch.cat([all_correct, control_texts], dim=0)
    all_imgs = torch.cat([imgs_wif, imgs_wwt, imgs_wnm, imgs_control], dim=0)
    
    # Single forward pass with image reconstruction
    all_probs, all_recon = model_forward_with_tokens(model, all_texts, all_imgs, ret_imgs=True)
    
    # DPO losses for task chunks
    task_total = sum(chunk_sizes[:3])
    task_dpo_losses = []
    task_accuracies = []
    offset = 0
    wrong_offset = 0
    for i in range(3):
        cs = chunk_sizes[i]
        dpo_loss_i, acc_i = get_dpo_text_loss(
            all_probs[offset:offset + cs, :, :],
            all_texts[offset:offset + cs],
            all_wrong[wrong_offset:wrong_offset + cs],
            all_lens[wrong_offset:wrong_offset + cs]
        )
        task_dpo_losses.append(dpo_loss_i)
        task_accuracies.append(acc_i)
        offset += cs
        wrong_offset += cs

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
              f"  {task_dpo_losses[0].item()} willIntersectForward (DPO+SFT), acc={task_accuracies[0]:.3f}\n"
              f"  {task_dpo_losses[1].item()} whichWayTurn (DPO+SFT), acc={task_accuracies[1]:.3f}\n"
              f"  {task_dpo_losses[2].item()} whatNextMove (DPO+SFT), acc={task_accuracies[2]:.3f}\n"
              f"  {control_loss.item()} control\n")

    if reset_model:
        model.reset()

    return (loss.item(), task_dpo_losses[0].item(), task_dpo_losses[1].item(), task_dpo_losses[2].item(), control_loss.item(), img_loss.item())


def relposition_qa_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if compute_grad:
        return _relposition_qa_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _relposition_qa_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
