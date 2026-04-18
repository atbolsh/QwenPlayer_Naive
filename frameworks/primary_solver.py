# Primary Solver framework
# Task: Given the current game state, predict the correct next move.
#
# Builds a prefix of the optimal forward-only solution, advances the game
# to the state right before the last kept move, and trains via DPO on
# that single "next move" token.

import random
from copy import deepcopy

from .general_framework import *
from .general_qa import STOP_TOKEN_ID, tensorify_list, get_lens

from .game_logic_solver import trace_forward

# Action code -> token string mapping (forward-only; no backward moves)
ACTION_TO_TOKEN = {1: '<forward>', 3: '<clock>', 4: '<anticlock>'}

# Resolve token IDs once at import time
forward_id = tokenizer.convert_tokens_to_ids('<forward>')
clock_id = tokenizer.convert_tokens_to_ids('<clock>')
anticlock_id = tokenizer.convert_tokens_to_ids('<anticlock>')

ACTION_TO_TOKEN_ID = {1: forward_id, 3: clock_id, 4: anticlock_id}
ALL_ACTION_IDS = [forward_id, clock_id, anticlock_id]

# Prompts - the empty string makes this framework's behaviour the default
prompts_primary_solver = ["Solve", "Solve this game", ""]
prompts_primary_solver_tensor = tensorify_list(prompts_primary_solver)
prompts_primary_solver_lens = get_lens(prompts_primary_solver_tensor)

MAX_MOVE_PREFIX = 32

########


def primary_solver_data(batch_size):
    """Generate training data for the primary solver.

    For each sample:
      1. Random game settings + full forward-only solution trace
      2. Random prefix length (1..min(32, trace_len))
      3. Advance game to state *before* last kept move -> capture image
      4. Build correct/wrong token sequences (differ only at last action pos)

    Returns:
        imgs:           (batch, 3, 224, 224) game screenshots
        correct_texts:  (batch, seq_len) token ids with correct next move
        wrong_texts:    (batch, seq_len) token ids with wrong next move
        loss_positions: (batch,) index of the token position to compute loss on
    """
    S = get_settings_batch(batch_size)
    prompt_num = prompts_primary_solver_tensor.size(0)
    prompt_size = prompts_primary_solver_tensor.size(1)

    total_len = prompt_size + MAX_MOVE_PREFIX + 2  # +1 space, +1 stop token headroom

    correct_tensor = torch.zeros((batch_size, total_len), device=device, dtype=prompts_primary_solver_tensor.dtype)
    wrong_tensor = torch.zeros((batch_size, total_len), device=device, dtype=prompts_primary_solver_tensor.dtype)
    loss_positions = torch.zeros(batch_size, dtype=torch.long, device=device)

    imgs = torch.zeros(batch_size, 224, 224, 3, dtype=torch.float32)

    for i in range(batch_size):
        trace = trace_forward(S[i])
        trace_len = len(trace)
        cut = random.randint(1, min(MAX_MOVE_PREFIX, trace_len))
        is_full = (cut == trace_len)

        # Advance game to state right before the last kept move
        G2 = discreteGame(deepcopy(S[i]))
        if is_full:
            # Play all moves; gold is consumed. Target = <|im_end|>
            for step_idx in range(trace_len):
                G2.actions[trace[step_idx]]()
        else:
            # Play moves 0..cut-2; state is right before move cut-1
            for step_idx in range(cut - 1):
                G2.actions[trace[step_idx]]()
        imgs[i] = torch.tensor(G2.getData(), dtype=torch.float32)

        # Pick a random prompt
        p_idx = random.randint(0, prompt_num - 1)
        prompt = prompts_primary_solver_tensor[p_idx]
        p_len = prompts_primary_solver_lens[p_idx]

        # Fill prompt portion
        correct_tensor[i, :prompt_size] = prompt
        wrong_tensor[i, :prompt_size] = prompt

        # Space token after prompt (mirrors _stitch convention)
        space_token_id = tokenizer.encode(' ', add_special_tokens=False)[0] if tokenizer.encode(' ', add_special_tokens=False) else 220
        correct_tensor[i, p_len] = space_token_id
        wrong_tensor[i, p_len] = space_token_id

        # Fill action tokens for moves 0..cut-1
        write_start = p_len + 1
        for j in range(cut):
            tok_id = ACTION_TO_TOKEN_ID[trace[j]]
            correct_tensor[i, write_start + j] = tok_id
            wrong_tensor[i, write_start + j] = tok_id

        if is_full:
            # Append <|im_end|> as the target after the last action
            target_pos = write_start + cut
            correct_tensor[i, target_pos] = STOP_TOKEN_ID
            # Wrong: a random action token instead of <|im_end|>
            wrong_token = random.choice(ALL_ACTION_IDS)
            wrong_tensor[i, target_pos] = wrong_token
            loss_positions[i] = target_pos
        else:
            # The last action token is the target; replace it in wrong_tensor
            target_pos = write_start + cut - 1
            correct_action = ACTION_TO_TOKEN_ID[trace[cut - 1]]
            wrong_action = random.choice([a for a in ALL_ACTION_IDS if a != correct_action])
            wrong_tensor[i, target_pos] = wrong_action
            loss_positions[i] = target_pos
            # No <|im_end|>; rest stays zero-padded

    imgs = torch.permute(imgs, (0, 3, 1, 2)).contiguous().to(device)
    return imgs, correct_tensor.contiguous(), wrong_tensor.contiguous(), loss_positions


def get_primary_solver_dpo_loss(logits, correct_texts, wrong_texts, loss_positions, beta=0.1):
    """DPO loss computed on a single token position per sample.

    Args:
        logits:         (batch, vocab, seq_len) model output from correct_texts
        correct_texts:  (batch, seq_len) token ids with correct next move
        wrong_texts:    (batch, seq_len) token ids with wrong next move
        loss_positions: (batch,) the position of the target token in the sequence
        beta:           DPO temperature
    """
    batch_size = logits.size(0)

    # Shifted logits: logits[:, :, t] predicts token at position t+1
    # So to get the prediction *for* position p, we read logits[:, :, p-1]
    pred_positions = loss_positions - 1  # (batch,)

    log_probs = F.log_softmax(logits, dim=1)  # (batch, vocab, seq_len)

    correct_targets = correct_texts[torch.arange(batch_size, device=logits.device), loss_positions]  # (batch,)
    wrong_targets = wrong_texts[torch.arange(batch_size, device=logits.device), loss_positions]      # (batch,)

    # Gather log-probs at the prediction position for correct and wrong tokens
    # log_probs[:, :, pred_pos] -> (batch, vocab) per sample
    pred_lp = log_probs[torch.arange(batch_size, device=logits.device), :, pred_positions]  # (batch, vocab)

    correct_lp = pred_lp[torch.arange(batch_size, device=logits.device), correct_targets]  # (batch,)
    wrong_lp = pred_lp[torch.arange(batch_size, device=logits.device), wrong_targets]      # (batch,)

    return -F.logsigmoid(beta * (correct_lp - wrong_lp)).mean()


########


def _primary_solver_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")

    if model_eval:
        model.pipe.model.eval()

    if training:
        model.pipe.model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")

    imgs, correct_texts, wrong_texts, loss_positions = primary_solver_data(batch_size)

    task_probs, task_recon = model_forward_with_tokens(model, correct_texts, imgs, ret_imgs=True)

    img_loss = img_criterion(task_recon, imgs)
    dpo_loss = get_primary_solver_dpo_loss(task_probs, correct_texts, wrong_texts, loss_positions)
    loss = img_loss + (dpo_loss / 5000)

    if training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.soft_reset()

    if printing:
        print(f"Total loss: {loss.item()}; that's {dpo_loss.item()} task (DPO) and {img_loss.item()} img loss\n")

    if reset_model:
        model.reset()

    return loss.item(), dpo_loss.item(), img_loss.item()


def primary_solver_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if compute_grad:
        return _primary_solver_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _primary_solver_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
