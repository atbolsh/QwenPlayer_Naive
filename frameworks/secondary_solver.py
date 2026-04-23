# Secondary Solver framework
# Task: Given the current game state, predict the correct next move.
#
# Identical to primary_solver, EXCEPT: in the tensor fed to the LLM, every
# action token in the move history is replaced with a random *different*
# action token, so that the degenerate "copy the previous move" strategy
# can no longer succeed. The single target action token at the loss
# position is preserved so that the SFT/DPO signal still trains the model
# to predict the true correct next move.
#
# The game simulation and rendered image are completely unaffected by this
# randomization: only the text tokens used as model input are shuffled.

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
ALL_ACTION_IDS_SET = set(ALL_ACTION_IDS)

# Prompts (separate list; empty-string prompt is kept to mirror primary_solver)
prompts_secondary_solver = ["Solve", "Solve this game", ""]
prompts_secondary_solver_tensor = tensorify_list(prompts_secondary_solver)
prompts_secondary_solver_lens = get_lens(prompts_secondary_solver_tensor)

MAX_MOVE_PREFIX = 32

########


def secondary_solver_data(batch_size):
    """Generate training data for the secondary solver.

    Same game/image generation as primary_solver, but action tokens
    preceding the target position are randomized in the input tensors
    (correct and wrong) so the model cannot cheat by copying the
    previous action token.

    Returns:
        imgs:           (batch, 3, 224, 224) game screenshots
        correct_texts:  (batch, seq_len) token ids with correct next move
                        (action history preceding target is randomized)
        wrong_texts:    (batch, seq_len) token ids with wrong next move
                        (same randomized history as correct_texts)
        loss_positions: (batch,) index of the token position to compute loss on
    """
    S = get_settings_batch(batch_size)
    prompt_num = prompts_secondary_solver_tensor.size(0)
    prompt_size = prompts_secondary_solver_tensor.size(1)

    total_len = prompt_size + MAX_MOVE_PREFIX + 2  # +1 space, +1 stop token headroom

    correct_tensor = torch.zeros((batch_size, total_len), device=device, dtype=prompts_secondary_solver_tensor.dtype)
    wrong_tensor = torch.zeros((batch_size, total_len), device=device, dtype=prompts_secondary_solver_tensor.dtype)
    loss_positions = torch.zeros(batch_size, dtype=torch.long, device=device)

    imgs = torch.zeros(batch_size, 224, 224, 3, dtype=torch.float32)

    for i in range(batch_size):
        trace = trace_forward(S[i])
        trace_len = len(trace)
        max_cut = min(MAX_MOVE_PREFIX, trace_len)
        # 20% chance: prompt alone (cut=1, predict first action)
        # 20% chance: full trace (cut=trace_len, predict <|im_end|>) — only if it fits
        # 60% chance (+ overflow from full trace): uniform over 1..max_cut
        r = random.random()
        if r < 0.2:
            cut = 1
        elif r < 0.4 and trace_len <= MAX_MOVE_PREFIX:
            cut = trace_len
        else:
            cut = random.randint(1, max_cut)
        is_full = (cut == trace_len)

        # Advance game to state right before the last kept move.
        # NOTE: This uses the REAL trace — the game simulation must stay
        # faithful so the rendered image reflects the true game state.
        G2 = discreteGame(deepcopy(S[i]))
        if is_full:
            for step_idx in range(trace_len):
                G2.actions[trace[step_idx]]()
        else:
            for step_idx in range(cut - 1):
                G2.actions[trace[step_idx]]()
        imgs[i] = torch.tensor(G2.getData(), dtype=torch.float32)

        # Pick a random prompt
        p_idx = random.randint(0, prompt_num - 1)
        prompt = prompts_secondary_solver_tensor[p_idx]
        p_len = prompts_secondary_solver_lens[p_idx]

        # Fill prompt portion
        correct_tensor[i, :prompt_size] = prompt
        wrong_tensor[i, :prompt_size] = prompt

        # Space token after prompt (mirrors _stitch convention)
        space_token_id = tokenizer.encode(' ', add_special_tokens=False)[0] if tokenizer.encode(' ', add_special_tokens=False) else 220
        correct_tensor[i, p_len] = space_token_id
        wrong_tensor[i, p_len] = space_token_id

        write_start = p_len + 1
        last_write_idx = write_start + cut - 1
        assert last_write_idx < total_len, (
            f"OOB writing actions: sample {i}, write_start={write_start}, cut={cut}, "
            f"last_write_idx={last_write_idx}, total_len={total_len}, "
            f"p_len={p_len}, trace_len={trace_len}, is_full={is_full}")

        # Compute the target position first so we know which action index
        # (if any) must be preserved rather than randomized.
        if is_full:
            target_pos = write_start + cut  # target is <|im_end|>; all action tokens randomized
        else:
            target_pos = write_start + cut - 1  # target IS the last action token; preserve it

        # Fill action tokens for moves 0..cut-1. For every action position
        # except the target, substitute a random *different* action id so
        # the model can't cheat by copying the previous move token.
        for j in range(cut):
            pos = write_start + j
            true_tok_id = ACTION_TO_TOKEN_ID[trace[j]]
            if pos == target_pos:
                # Preserve the real target action so the SFT/DPO signal
                # is still meaningful (non-full case only).
                tok_id = true_tok_id
            else:
                # Random other action id (any of the 3 actions except the true one).
                tok_id = random.choice([a for a in ALL_ACTION_IDS if a != true_tok_id])
            correct_tensor[i, pos] = tok_id
            wrong_tensor[i, pos] = tok_id

        if is_full:
            # Append <|im_end|> as the target after the last action
            assert target_pos < total_len, (
                f"OOB full-trace target: sample {i}, target_pos={target_pos}, total_len={total_len}, "
                f"write_start={write_start}, cut={cut}, p_len={p_len}, trace_len={trace_len}")
            correct_tensor[i, target_pos] = STOP_TOKEN_ID
            wrong_token = random.choice(ALL_ACTION_IDS)
            wrong_tensor[i, target_pos] = wrong_token
            loss_positions[i] = target_pos
        else:
            # The last action token is the target; replace it in wrong_tensor
            assert target_pos < total_len, (
                f"OOB non-full target: sample {i}, target_pos={target_pos}, total_len={total_len}, "
                f"write_start={write_start}, cut={cut}, p_len={p_len}, trace_len={trace_len}")
            correct_action = ACTION_TO_TOKEN_ID[trace[cut - 1]]
            wrong_action = random.choice([a for a in ALL_ACTION_IDS if a != correct_action])
            # correct_tensor[i, target_pos] was already set to the true action above.
            wrong_tensor[i, target_pos] = wrong_action
            loss_positions[i] = target_pos

    max_lp = loss_positions.max().item()
    assert max_lp < total_len, (
        f"secondary_solver_data: loss_positions max={max_lp} >= total_len={total_len}")

    imgs = torch.permute(imgs, (0, 3, 1, 2)).contiguous().to(device)
    return imgs, correct_tensor.contiguous(), wrong_tensor.contiguous(), loss_positions


def get_secondary_solver_dpo_loss(logits, correct_texts, wrong_texts, loss_positions, beta=0.1, sft_weight=2.0):
    """DPO + SFT loss computed on a single token position per sample.

    Identical to the primary_solver loss; duplicated here so the secondary
    framework stands on its own.
    """
    batch_size = logits.size(0)
    seq_len_logits = logits.size(2)
    seq_len_texts = correct_texts.size(1)
    max_lp = loss_positions.max().item()
    min_lp = loss_positions.min().item()

    assert seq_len_logits == seq_len_texts, (
        f"get_secondary_solver_dpo_loss: logits seq_len={seq_len_logits} != "
        f"correct_texts seq_len={seq_len_texts}")
    assert max_lp < seq_len_texts, (
        f"get_secondary_solver_dpo_loss: max loss_position={max_lp} >= "
        f"correct_texts dim1={seq_len_texts}. loss_positions={loss_positions.tolist()}")
    assert min_lp >= 1, (
        f"get_secondary_solver_dpo_loss: min loss_position={min_lp} < 1 "
        f"(pred_position would be {min_lp - 1}). loss_positions={loss_positions.tolist()}")

    batch_idx = torch.arange(batch_size, device=logits.device)

    pred_positions = loss_positions - 1  # (batch,)

    log_probs = F.log_softmax(logits, dim=1)  # (batch, vocab, seq_len)

    correct_targets = correct_texts[batch_idx, loss_positions]
    wrong_targets = wrong_texts[batch_idx, loss_positions]

    pred_lp = log_probs[batch_idx, :, pred_positions]

    correct_lp = pred_lp[batch_idx, correct_targets]
    wrong_lp = pred_lp[batch_idx, wrong_targets]

    diff_mask = (correct_targets != wrong_targets).float()
    dpo_per_sample = -F.logsigmoid(beta * (correct_lp - wrong_lp))
    dpo_loss = (dpo_per_sample * diff_mask).sum() / diff_mask.sum().clamp(min=1)

    pred_logits = logits[batch_idx, :, pred_positions]
    sft_loss = F.cross_entropy(pred_logits, correct_targets)

    argmax_tokens = pred_logits.argmax(dim=1)
    accuracy = (argmax_tokens == correct_targets).float().mean()

    return dpo_loss + sft_weight * sft_loss, accuracy.item()


########


def _secondary_solver_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")

    if model_eval:
        model.pipe.model.eval()

    if training:
        model.pipe.model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")

    imgs, correct_texts, wrong_texts, loss_positions = secondary_solver_data(batch_size)

    assert correct_texts.size(1) == wrong_texts.size(1), (
        f"secondary_solver_batch: correct_texts seq_len={correct_texts.size(1)} != "
        f"wrong_texts seq_len={wrong_texts.size(1)}")

    task_probs, task_recon = model_forward_with_tokens(model, correct_texts, imgs, ret_imgs=True)

    assert task_probs.size(2) == correct_texts.size(1), (
        f"secondary_solver_batch: model returned logits with seq_len={task_probs.size(2)} "
        f"but correct_texts has seq_len={correct_texts.size(1)}. "
        f"task_probs shape={tuple(task_probs.shape)}, "
        f"correct_texts shape={tuple(correct_texts.shape)}")

    img_loss = img_criterion(task_recon, imgs)
    dpo_loss, accuracy = get_secondary_solver_dpo_loss(task_probs, correct_texts, wrong_texts, loss_positions)
    loss = img_loss + (dpo_loss / 5000)

    if training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.soft_reset()

    if printing:
        print(f"Total loss: {loss.item()}; that's {dpo_loss.item()} task (DPO+SFT) and {img_loss.item()} img loss\n"
              f"  correct answer accuracy: {accuracy:.3f}\n")

    if reset_model:
        model.reset()

    return loss.item(), dpo_loss.item(), img_loss.item()


def secondary_solver_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if compute_grad:
        return _secondary_solver_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _secondary_solver_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
