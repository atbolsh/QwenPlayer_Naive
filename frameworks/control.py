# Control framework - basic image reconstruction and text prediction training

from .general_framework import *

def _control_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.pipe.model.eval()

    if training:
        model.pipe.model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")

    img_set = G.random_full_image_set(restrict_angles=True)

    np_b_size = img_set.shape[0]

    if batch_size < np_b_size:
        inds = random.sample(list(range(np_b_size)), batch_size)
        img_set = img_set[inds]

    if batch_size > np_b_size:
        while img_set.shape[0] < batch_size:
            img_set = np.concatenate((img_set, G.random_full_image_set(restrict_angles=True)))
        if img_set.shape[0] > batch_size:
            inds = random.sample(list(range(img_set.shape[0])), batch_size)
            img_set = img_set[inds]

    img_tensor = torch.permute(torch.FloatTensor(img_set).to(device), (0, 3, 1, 2))

    ind = (batch_num * batch_size) % num_controls
    if ind + batch_size > num_controls:
        ind = num_controls - batch_size
    control_texts = get_text_batch(sdt, ind, batch_size)

    # Use adapter function for QwenAgentPlayer
    control_probs, control_recon = model_forward_with_tokens(model, control_texts, img_tensor, ret_imgs=True)

    img_loss = img_criterion(control_recon, img_tensor)
    text_loss = get_text_loss(control_probs, control_texts)

    loss = img_loss + (text_loss / 1000)

    if training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.soft_reset()

    if printing:
        print(f"Total recon loss: {loss.item()}; that's {text_loss.item()} text and {img_loss.item()} img\n\n")

    if reset_model:
        model.reset()

    return loss.item(), text_loss.item(), img_loss.item()


def control_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if compute_grad:
        return _control_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _control_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
