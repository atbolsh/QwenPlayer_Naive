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

task2_Lreplies_lrgold = ["Left", "It's to the left.", "It's on the left.", "Go left."]
task2_Rreplies_lrgold = ["Right", "It's to the right.", "It's on the right.", "Go right."]

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

task2_Ureplies_udgold = ["Up", "Above", "It's up.", "It's above me.", "Go up."]
task2_Dreplies_udgold = ["Down", "Below", "It's down.", "It's below me.", "Go down."]

# Left-right agent position
task2_prompts_lragent = [
    "Are you to the left or right of the gold?",
    "Which side is the gold on?",
    "Is the agent to the left or right of the gold?",
    "Please tell me whether you are right or left of the gold.",
    "Please tell me which side you are relative to the gold.",
    "On which side of the gold are you?"
]

task2_Lreplies_lragent = ["Left", "I'm to the left.", "The agent is on the left."]
task2_Rreplies_lragent = ["Right", "I'm to the right.", "The agent is on the right."]

# Up-down agent position
task2_prompts_udagent = [
    "Are you below or above the gold?",
    "Is the agent above or below the gold?",
    "Please tell me whether you are up or down from the gold.",
    "Please tell me whether you are above or below the gold."
]

task2_Ureplies_udagent = ["Up", "I'm above it.", "The agent is above the gold."]
task2_Dreplies_udagent = ["Down", "I'm below it.", "The agent is below the gold."]

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
# Text generators (simple versions for training)

task2_lrgold_generator_simple = lambda settings_batch: text_generator_simple(
    settings_batch, task2_prompts_lrgold_tensor, task2_Lreplies_lrgold_tensor,
    task2_Rreplies_lrgold_tensor, task2_prompts_lrgold_lens, is_gold_left, device)

task2_udgold_generator_simple = lambda settings_batch: text_generator_simple(
    settings_batch, task2_prompts_udgold_tensor, task2_Ureplies_udgold_tensor,
    task2_Dreplies_udgold_tensor, task2_prompts_udgold_lens, is_gold_up, device)

task2_lragent_generator_simple = lambda settings_batch: text_generator_simple(
    settings_batch, task2_prompts_lragent_tensor, task2_Lreplies_lragent_tensor,
    task2_Rreplies_lragent_tensor, task2_prompts_lragent_lens, is_agent_left, device)

task2_udagent_generator_simple = lambda settings_batch: text_generator_simple(
    settings_batch, task2_prompts_udagent_tensor, task2_Ureplies_udagent_tensor,
    task2_Dreplies_udagent_tensor, task2_prompts_udagent_lens, is_agent_up, device)

text_generators_simple = [
    task2_lrgold_generator_simple,
    task2_udgold_generator_simple,
    task2_lragent_generator_simple,
    task2_udagent_generator_simple
]

########

def _qa_task_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
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
    
    texts_lrg = task2_lrgold_generator_simple(S)
    texts_udg = task2_udgold_generator_simple(S)
    texts_lra = task2_lragent_generator_simple(S)
    texts_uda = task2_udagent_generator_simple(S)

    ind = (batch_num * batch_size) % num_controls
    if ind + batch_size > num_controls:
        ind = num_controls - batch_size
    control_texts = get_text_batch(sdt, ind, batch_size)

    all_texts = [control_texts, texts_lrg, texts_udg, texts_lra, texts_uda]
    text_inds = list(range(5))
    
    if random_order:
        random.shuffle(text_inds)

    all_probs = [0 for _ in text_inds]
    
    # Process each text batch
    for i, t_ind in enumerate(text_inds):
        all_probs[t_ind] = model_forward_with_tokens(model, all_texts[t_ind], imgs, ret_imgs=False)

    all_losses = [get_text_loss(all_probs[i], all_texts[i]) for i in range(5)]
    loss = sum(all_losses)

    if training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.soft_reset()

    if printing:
        print(f"Total loss: {loss.item()}:\n{all_losses[0].item()} control,\n{all_losses[1].item()} lrg,\n{all_losses[2].item()} udg,\n{all_losses[3].item()} lra,\n{all_losses[4].item()} uda\n\n")

    if reset_model:
        model.reset()

    return (loss.item(), all_losses[0].item(), all_losses[1].item(), all_losses[2].item(), all_losses[3].item(), all_losses[4].item())


def qa_task_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if compute_grad:
        return _qa_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _qa_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
