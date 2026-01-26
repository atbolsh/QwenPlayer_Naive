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

Yreplies_willIntersectForward = ["Yep", "Absolutely.", "Certainly", "I think so.", "Uh-huh.", "Sure"]
Nreplies_willIntersectForward = ["Nuh-uh", "No", "I don't think so.", "Certainly not", "Absolutely not", "Nah"]

prompts_whichWayTurn = [
    "Which way should you turn, do you figure?",
    "Damn, how can I twist in the right direction?",
    "Which way to fix our direction?",
    "How should you turn?"
]

CWreplies_whichWayTurn = ["Clockwise", "I should turn clockwise", "CW", "Clockwise, sir!"]
CCWreplies_whichWayTurn = ["Counter-clockwise", "I should turn counter-clockwise", "CCW", "Counter-clockwise, sir!"]

prompts_whatNextMove = [
    "Damn it, what's the move here, partner?",
    "What should you do here?",
    "In this position, what should you do?",
    "How do you figure, what's the move for us?",
    "What's the move?"
]

Freplies_whatNextMove = ["Just go straight.", "We just go straight", "Full speed ahead!"]
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

willIntersectForward_generator_simple = lambda settings_batch: text_generator_simple(
    settings_batch, prompts_willIntersectForward_tensor, Yreplies_willIntersectForward_tensor,
    Nreplies_willIntersectForward_tensor, prompts_willIntersectForward_lens, willIntersectForward, device
)

whichWayTurn_generator_simple = lambda settings_batch: text_generator_simple(
    settings_batch, prompts_whichWayTurn_tensor, CWreplies_whichWayTurn_tensor,
    CCWreplies_whichWayTurn_tensor, prompts_whichWayTurn_lens, best_turn_cw, device
)

whatNextMove_generator_simple = lambda settings_batch: text_generator_simple_GENERAL(
    settings_batch, prompts_whatNextMove_tensor,
    [Freplies_whatNextMove_tensor, CWreplies_whatNextMove_tensor, CCWreplies_whatNextMove_tensor],
    prompts_whatNextMove_lens, best_move, device
)

########

def _relposition_qa_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.pipe.model.eval()

    if training:
        model.pipe.model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")
        
    S = get_settings_batch(batch_size)
    imgs = get_images(S)
    
    texts_wif = willIntersectForward_generator_simple(S)
    texts_wwt = whichWayTurn_generator_simple(S)
    texts_wnm = whatNextMove_generator_simple(S)

    ind = (batch_num * batch_size) % num_controls
    if ind + batch_size > num_controls:
        ind = num_controls - batch_size
    control_texts = get_text_batch(sdt, ind, batch_size)

    all_texts = [control_texts, texts_wif, texts_wwt, texts_wnm]
    text_inds = list(range(4))
    
    if random_order:
        random.shuffle(text_inds)

    all_probs = [0 for _ in text_inds]
    for t_ind in text_inds:
        all_probs[t_ind] = model_forward_with_tokens(model, all_texts[t_ind], imgs, ret_imgs=False)

    all_losses = [get_text_loss(all_probs[i], all_texts[i]) for i in range(4)]
    loss = sum(all_losses)

    if training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.soft_reset()

    if printing:
        print(f"Total loss: {loss.item()}:\n{all_losses[0].item()} control,\n{all_losses[1].item()} willIntersectForward,\n{all_losses[2].item()} whichWayTurn,\n{all_losses[3].item()} whatNextMove\n\n")

    if reset_model:
        model.reset()

    return (loss.item(), all_losses[0].item(), all_losses[1].item(), all_losses[2].item(), all_losses[3].item())


def relposition_qa_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if compute_grad:
        return _relposition_qa_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _relposition_qa_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
