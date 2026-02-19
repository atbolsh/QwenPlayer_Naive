# Memory Canvas Use framework
# Task: Recall images from canvas history (uses existing canvases, not fresh images)
# The model is asked to reconstruct an image it has already seen (from canvases or current input).
# Canvas order in image_encodings: [oldest_canvas, ..., newest_canvas, current_input]

from .general_framework import *

# Max lookback: 3 canvases + 1 current = 4 possible images
MAX_LOOKBACK = 4

# Regularization constants for VisionWeightedSum CE training
WEIGHT_CE_LABEL_SMOOTHING = 0.01
WEIGHT_CE_LOGIT_CLAMP = 10.0
WEIGHT_CE_FLOOD_LEVEL = 0.01
WEIGHT_CLAMP_RANGE = 2.0

current_image_prompts = [
    "Hey, recall the current image again?",
    "Focus on the present view, please.",
    "What do you see right now?",
    "Think about the present game, for a moment."
]

prev_image_prompts = [
    "Hey, recall the last image, again?",
    "Focus on the last view, please.",
    "What did you see 1 image ago?",
    "Think about the last game, for a moment.",
    "Woah! What was that 1 image ago, again???"
]

def get_image_prompts(ind):
    return [
        f"Hey, could you recall the image {ind} ago, again?",
        f"Focus on the view {ind} ago, please.",
        f"Hey, recall the game {ind} ago for me, will you?",
        f"Woah! What was that {ind} images ago, again??",
        f"Think about the game from {ind} ago for me, will you?",
        f"What did you see {ind} games ago, again.",
        f"Think about the view {ind} steps ago, for a moment."
    ]

# lookback_prompts[k] = prompts for recalling the image k steps ago
# k=0: current image, k=1: previous, k=2: two ago, k=3: three ago
lookback_prompts = [current_image_prompts, prev_image_prompts]
for n in range(2, MAX_LOOKBACK):
    lookback_prompts.append(get_image_prompts(n))


def mem_task_img_sample(num_sample=40):
    """Generate random game images for the current input."""
    img_in = torch.zeros(num_sample, 224, 224, 3, dtype=torch.float32)
    for i in range(num_sample):
        bare_settings = G.random_bare_settings(gameSize=224, max_agent_offset=2.0)
        G2 = discreteGame(bare_settings)
        img_in[i] = torch.tensor(G2.getData(), dtype=torch.float32)
    img_in = torch.permute(img_in, (0, 3, 1, 2)).contiguous().to(device)
    return img_in


def _mem_canvas_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True,
                      model_eval=True, reset_model=True, printing=True, training=False,
                      use_lora=False, train_weights=True, only_weight_ce=False,
                      grad_clip_norm=1.0):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")

    if model_eval:
        model.pipe.model.eval()

    if training:
        model.pipe.model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")

    if only_weight_ce and not train_weights:
        raise ValueError("only_weight_ce requires train_weights=True")

    # How many images are available: existing canvases + the current input we'll provide
    num_canvases = len(model.canvases)
    num_available = num_canvases + 1  # canvases + current input image

    # Generate a fresh input image batch
    input_imgs = mem_task_img_sample(batch_size)

    # For each batch element, pick a random lookback k in [0, num_available)
    # k=0 means "recall current input", k=1 means "recall newest canvas", etc.
    lookback_vals = np.random.randint(0, num_available, (batch_size,))

    # Build target images and prompts per batch element
    target_weight_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
    prompts = []

    if not only_weight_ce:
        target = torch.zeros_like(input_imgs)

    for i in range(batch_size):
        k = lookback_vals[i]
        if not only_weight_ce:
            if k == 0:
                target[i] = input_imgs[i].detach()
            else:
                target[i] = model.canvases[-k][i].detach()

        prompts.append(random.choice(lookback_prompts[k]))
        target_weight_indices[i] = num_available - 1 - k

    if not only_weight_ce:
        target = target.contiguous()

    prompt_tensor = encode_batch(prompts).contiguous().to(device)
    padded_prompt_tensor = torch.zeros((batch_size, MAX_SEQ_LENGTH), dtype=prompt_tensor.dtype, device=device)
    padded_prompt_tensor[:, :prompt_tensor.size(1)] += prompt_tensor

    # Forward pass
    task_img_loss_val = 0.0
    task_text_loss_val = 0.0
    weight_ce_loss_val = 0.0

    if only_weight_ce:
        # Still generate images so they get stored as canvases for future batches
        text_probs, recon, canvas_logits = model_forward_with_tokens(
            model, padded_prompt_tensor, input_imgs, ret_imgs=True,
            return_canvas_logits=True,
        )
    elif train_weights:
        text_probs, recon, canvas_logits = model_forward_with_tokens(
            model, padded_prompt_tensor, input_imgs, ret_imgs=True,
            return_canvas_logits=True,
        )
    else:
        text_probs, recon = model_forward_with_tokens(
            model, padded_prompt_tensor, input_imgs, ret_imgs=True,
        )

    # Compute losses
    if only_weight_ce:
        loss = torch.tensor(0.0, device=device)
    else:
        task_img_loss = img_criterion(recon, target)
        task_text_loss = get_text_loss(text_probs, padded_prompt_tensor)
        loss = task_img_loss + (task_text_loss / 1000)
        task_img_loss_val = task_img_loss.item()
        task_text_loss_val = task_text_loss.item()

    if train_weights and canvas_logits is not None:
        # Logit clamping: prevent extreme values
        clamped_logits = canvas_logits.squeeze(-1).clamp(-WEIGHT_CE_LOGIT_CLAMP, WEIGHT_CE_LOGIT_CLAMP)
        # Label smoothing: prevents driving logits to infinity
        weight_ce_loss = F.cross_entropy(clamped_logits, target_weight_indices,
                                         label_smoothing=WEIGHT_CE_LABEL_SMOOTHING)
        # Loss flooding: prevent over-optimization below flood level
        weight_ce_loss = torch.abs(weight_ce_loss - WEIGHT_CE_FLOOD_LEVEL) + WEIGHT_CE_FLOOD_LEVEL
        weight_ce_loss_val = weight_ce_loss.item()

        if only_weight_ce:
            loss = weight_ce_loss
        else:
            loss = loss + (weight_ce_loss / 1000)

    if training:
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.pipe.model.parameters() if p.requires_grad],
                max_norm=grad_clip_norm,
            )
        optimizer.step()
        optimizer.zero_grad()
        # Weight clamping on img_weight parameters
        with torch.no_grad():
            for p in model.pipe.model.img_weight.parameters():
                p.clamp_(-WEIGHT_CLAMP_RANGE, WEIGHT_CLAMP_RANGE)
        model.soft_reset()

    if printing:
        msg = (f"Total loss: {loss.item():.4f}; "
               f"img_recall={task_img_loss_val:.4f}, "
               f"text={task_text_loss_val:.4f}, "
               f"num_canvases={num_canvases}")
        if train_weights:
            msg += f", weight_ce={weight_ce_loss_val:.4f}"
        print(msg + "\n")

    if reset_model:
        model.reset()

    return loss.item(), task_img_loss_val, task_text_loss_val, weight_ce_loss_val


def mem_canvas_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False,
                     random_order=True, model_eval=True, reset_model=True, printing=True,
                     training=False, use_lora=False, train_weights=True,
                     only_weight_ce=False, grad_clip_norm=1.0):
    if compute_grad:
        return _mem_canvas_batch(batch_size, model, optimizer, batch_num, random_order,
                                 model_eval, reset_model, printing, training, use_lora,
                                 train_weights=train_weights, only_weight_ce=only_weight_ce,
                                 grad_clip_norm=grad_clip_norm)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _mem_canvas_batch(batch_size, model, optimizer, batch_num, random_order,
                                     model_eval, reset_model, printing, training, use_lora,
                                     train_weights=train_weights, only_weight_ce=only_weight_ce,
                                     grad_clip_norm=grad_clip_norm)
