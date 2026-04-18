# Comparison V1 framework
# Task: Compare two game states and pick the better one

from .general_framework import *
from .general_qa import *
from .game_logic_solver import trace_forward

initializations_comparisonv1 = [
    "Wait for the full task."
]

initializations_comparisonv1_tensor = append_stop_token(tensorify_list(initializations_comparisonv1))

prompts_comparisonv1 = [
    "Is the first or the second better, do you think?",
    "Are you closer to the gold in the first or second game?",
    "Pick your game: first or second?",
    "Which of these do you prefer?"
]

prompts_comparisonv1_tensor = tensorify_list(prompts_comparisonv1)
prompts_comparisonv1_lens = get_lens(prompts_comparisonv1_tensor)

FirstReplies_comparisonv1 = ["First"]

SecondReplies_comparisonv1 = ["Second"]

FirstReplies_comparisonv1_tensor = tensorify_list(FirstReplies_comparisonv1)
SecondReplies_comparisonv1_tensor = tensorify_list(SecondReplies_comparisonv1)

########

def comparisonv1_data(batch_size):
    S1 = get_settings_batch(batch_size) 
    S2 = get_settings_batch(batch_size)
    
    imgs1 = get_images(S1)
    imgs2 = get_images(S2)

    texts1 = simple_sample(batch_size, initializations_comparisonv1_tensor)

    deciderDict = {}
    for i in range(batch_size):
        wait1 = len(trace_forward(S1[i]))
        wait2 = len(trace_forward(S2[i]))
        deciderDict[S1[i]] = (wait1 <= wait2)

    deciderFunc = lambda s: deciderDict[s]

    correct_texts, wrong_texts, prompt_lens = text_generator_dpo(
        S1, prompts_comparisonv1_tensor, FirstReplies_comparisonv1_tensor,
        SecondReplies_comparisonv1_tensor, prompts_comparisonv1_lens, deciderFunc, device
    )

    return imgs1, texts1, imgs2, correct_texts, wrong_texts, prompt_lens


def _comparisonv1_task_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.pipe.model.eval()

    if training:
        model.pipe.model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")

    imgs1, init_texts, imgs2, correct_texts, wrong_texts, prompt_lens = comparisonv1_data(batch_size)

    init_probs, init_recon = model_forward_with_tokens(model, init_texts, imgs1, ret_imgs=True)
    task_probs, task_recon = model_forward_with_tokens(model, correct_texts, imgs2, ret_imgs=True)

    l1 = img_criterion(init_recon, imgs1)
    l2 = img_criterion(task_recon, imgs2)
    img_loss = l1 + l2
    tl1 = get_text_loss(init_probs, init_texts)
    tl2 = get_dpo_text_loss(task_probs, correct_texts, wrong_texts, prompt_lens)
    text_loss = tl1 + tl2
    loss = img_loss + (text_loss / 5000)

    if training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.soft_reset()
    
    if printing:
        print(f"Total loss: {loss.item()}; that's {tl2.item()} task (DPO) and {tl1.item()} initialization text loss and {img_loss.item()} total img loss\n\n")

    if reset_model:
        model.reset()

    return loss.item(), l1.item(), l2.item(), tl1.item(), tl2.item()


def comparisonv1_task_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if compute_grad:
        return _comparisonv1_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _comparisonv1_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
