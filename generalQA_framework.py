from general_framework import *

# see tutorialQA_framework to see how to use these

def tensorify_list(text_list, device=device):
    return torch.tensor([x.ids for x in tokenizer.encode_batch(text_list)]).contiguous().to(device)

########

def get_lens(prompt_tensor):
    lens = []
    max_len = prompt_tensor.size()[1]
    for prompt in prompt_tensor:
        for ending in range(max_len):
            if prompt[ending] == 2:
                lens.append(ending) # want to skip the final </s>
                break
    return lens

########

# samples batchsize prompts from tensor 'prompts', with replacement
def simple_sample(batchsize, prompts, device=device):
    prompt_num, _ = prompts.size()
    inds = torch.randint(0, prompt_num, (batchsize,))
    return prompts[inds]

########

def _stitch(prompt, reply, container, length):
    container[:prompt.size()[0]] = prompt
    container[length] = 225 # tokenizer.encode(' ')
    reply_len = reply.size()[0] - 1
    container[length+1:length+reply_len+1] = reply[1:]
    return container

def text_generator(settings_batch, prompts, yes_responses, no_responses, prompt_lengths, func, device=device):
    batchsize = len(settings_batch)
    prompt_num, prompt_size = prompts.size()
    reply_size = max(yes_responses.size()[1], no_responses.size()[1])# - 1 # skip initial <s> in reply
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
    #input_tensor = torch.zeros((batchsize, reply_size + prompt_size), device=device, dtype=prompts.dtype)
    output_tensor = torch.zeros((batchsize, reply_size + prompt_size), device=device, dtype=prompts.dtype)
    
    for i in range(batchsize):
        ind = torch.randint(0, prompt_num, (1,)).item()
        prompt = prompts[ind]
        length = prompt_lengths[ind]
        if func(settings_batch[i]):
            reply = yes_responses[torch.randint(0, yes_num, (1,)).item()]
        else:
            reply = no_responses[torch.randint(0, no_num, (1,)).item()]
        #input_tensor[i] = prompt
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


