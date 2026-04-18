# General QA framework utilities
# Provides text generation helpers for QA-style training tasks

from .general_framework import *

STOP_TOKEN_ID = 151645  # <|im_end|>


def append_stop_token(tensor):
    """Replace the first padding position in each row with the stop token.

    Works on tensors produced by ``tensorify_list`` / ``encode_batch``.
    If a row has no padding (entirely filled), it is left unchanged.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    result = tensor.clone()
    for i in range(result.size(0)):
        for j in range(result.size(1)):
            tok = result[i, j].item()
            if tok == pad_id or tok == 0:
                result[i, j] = STOP_TOKEN_ID
                break
    return result

def tensorify_list(text_list, device=device):
    """Convert a list of text strings to a tensor of token ids using Qwen tokenizer."""
    encoded = tokenizer(
        text_list,
        padding='max_length',
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors='pt'
    )
    return encoded['input_ids'].contiguous().to(device)

########

def get_lens(prompt_tensor):
    """Get the lengths of each prompt (position of EOS token)."""
    lens = []
    max_len = prompt_tensor.size()[1]
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
    
    for prompt in prompt_tensor:
        found = False
        for ending in range(max_len):
            if prompt[ending] == eos_token_id:
                lens.append(ending)  # want to skip the final EOS
                found = True
                break
        if not found:
            lens.append(max_len)  # No EOS found, use full length
    return lens

########

# samples batchsize prompts from tensor 'prompts', with replacement
def simple_sample(batchsize, prompts, device=device):
    prompt_num, _ = prompts.size()
    inds = torch.randint(0, prompt_num, (batchsize,))
    return prompts[inds]

########

def _stitch(prompt, reply, container, length):
    """Stitch a prompt and reply together into a container tensor, with stop token."""
    container[:prompt.size()[0]] = prompt
    # Use a space token - get from tokenizer
    space_token_id = tokenizer.encode(' ', add_special_tokens=False)[0] if tokenizer.encode(' ', add_special_tokens=False) else 220
    container[length] = space_token_id
    reply_len = reply.size()[0] - 1
    container[length+1:length+reply_len+1] = reply[1:]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    stop_pos = length + 1
    for j in range(reply_len):
        tok = reply[1 + j].item()
        if tok == pad_id or tok == 0 or tok == STOP_TOKEN_ID:
            break
        stop_pos += 1
    if stop_pos < container.size(0):
        container[stop_pos] = STOP_TOKEN_ID

    return container

def text_generator(settings_batch, prompts, yes_responses, no_responses, prompt_lengths, func, device=device):
    batchsize = len(settings_batch)
    prompt_num, prompt_size = prompts.size()
    reply_size = max(yes_responses.size()[1], no_responses.size()[1])  # - 1 # skip initial <s> in reply
    yes_num = yes_responses.size()[0]
    no_num = no_responses.size()[0]
    input_tensor = torch.zeros((batchsize, reply_size + prompt_size), device=device, dtype=prompts.dtype)
    output_tensor = torch.zeros((batchsize, reply_size + prompt_size), device=device, dtype=prompts.dtype)
    
    for i in range(batchsize):
        ind = torch.randint(0, prompt_num, (1,)).item()
        prompt = prompts[ind]
        length = prompt_lengths[ind]
        if func(settings_batch[i]):
            reply = yes_responses[torch.randint(0, yes_num, (1,)).item()]
        else:
            reply = no_responses[torch.randint(0, no_num, (1,)).item()]
        input_tensor[i] = prompt
        _stitch(prompt, reply, output_tensor[i], length)

    return input_tensor.contiguous(), output_tensor.contiguous()

def text_generator_simple(settings_batch, prompts, yes_responses, no_responses, prompt_lengths, func, device=device):
    batchsize = len(settings_batch)
    prompt_num, prompt_size = prompts.size()
    reply_size = max(yes_responses.size()[1], no_responses.size()[1])
    yes_num = yes_responses.size()[0]
    no_num = no_responses.size()[0]
    output_tensor = torch.zeros((batchsize, reply_size + prompt_size), device=device, dtype=prompts.dtype)
    
    for i in range(batchsize):
        ind = torch.randint(0, prompt_num, (1,)).item()
        prompt = prompts[ind]
        length = prompt_lengths[ind]
        if func(settings_batch[i]):
            reply = yes_responses[torch.randint(0, yes_num, (1,)).item()]
        else:
            reply = no_responses[torch.randint(0, no_num, (1,)).item()]
        _stitch(prompt, reply, output_tensor[i], length)

    return output_tensor.contiguous()

# in this case, the func outputs an integer index for the right reply, not a boolean
def text_generator_simple_GENERAL(settings_batch, prompts, ordered_responses_list, prompt_lengths, func, device=device):
    batchsize = len(settings_batch)
    prompt_num, prompt_size = prompts.size()
    reply_size = max([x.size()[1] for x in ordered_responses_list])
    reply_nums = [x.size()[0] for x in ordered_responses_list]
    output_tensor = torch.zeros((batchsize, reply_size + prompt_size), device=device, dtype=prompts.dtype)
    
    for i in range(batchsize):
        ind = torch.randint(0, prompt_num, (1,)).item()
        prompt = prompts[ind]
        length = prompt_lengths[ind]
        reply_ind = func(settings_batch[i])
        reply = ordered_responses_list[reply_ind][torch.randint(0, reply_nums[reply_ind], (1,)).item()]
        _stitch(prompt, reply, output_tensor[i], length)

    return output_tensor.contiguous()
