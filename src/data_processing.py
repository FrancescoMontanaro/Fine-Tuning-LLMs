import torch
from typing import Optional
from transformers import TextStreamer


def build_chat(
    tokenizer, 
    max_length: int,
    user_text: str,
    answer_text: str,
    system_text: Optional[str] = None,
    enable_thinking: bool = False
) -> tuple[list[int], list[int], list[int]]:
    """
    Build chat input IDs, attention masks, and labels for fine-tuning.
    
    Args:
        tokenizer: The tokenizer to use.
        max_length (int): The maximum length of the input sequence.
        user_text (str): The user's input text.
        answer_text (str): The assistant's response text.
        system_text (Optional[str]): The system prompt text. Defaults to None.
        enable_thinking (bool): Whether to enable thinking mode. Defaults to False.
        
    Returns:
        tuple[list[int], list[int], list[int]]: input IDs, attention masks, and labels.
    """
    
    # Build the chat conversation
    prompts = [
        {"role": "system", "content": system_text} if system_text else None,
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": answer_text},
    ]

    conversation = [prompt for prompt in prompts if prompt]
    
    # Apply the chat template to full conversation
    full_text = tokenizer.apply_chat_template(
        conversation, 
        add_generation_prompt = False, 
        tokenize = True,
        return_tensors = None,
        enable_thinking = enable_thinking
    )
    
    # Build prompt only
    prompt_only = [prompt for prompt in prompts[:2] if prompt]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_only, 
        add_generation_prompt = True,
        tokenize = True,
        return_tensors = None,
        enable_thinking = enable_thinking
    )
    
    # Truncate if necessary
    if len(full_text) > max_length:
        full_text = full_text[:max_length]
    
    # Pad to max_length
    padding_length = max_length - len(full_text)
    input_ids = full_text + [tokenizer.pad_token_id] * padding_length
    attn = [1] * len(full_text) + [0] * padding_length
    
    # Create labels
    labels = [-100] * max_length

    # The response starts after the prompt
    prompt_length = len(prompt_ids)

    # Copy only the part of the response (non-padding)
    for i in range(prompt_length, len(full_text)):
        labels[i] = input_ids[i]
    
    # Return the input IDs, attention masks, and labels
    return input_ids, attn, labels


def generate_response(
    model, 
    tokenizer,
    user_message: str, 
    system_message: Optional[str] = None,
    max_new_tokens: int = 256,
    stream: bool = False,
    enable_thinking:  bool = False,
    **kwargs
) -> str:
    """
    Generate a response from the model given a user message and an optional system message.
    
    Args:
        model: The language model to use for generation.
        tokenizer: The tokenizer associated with the model.
        user_message (str): The user's message to which the model should respond.
        system_message (Optional[str]): An optional system message to provide context.
        max_new_tokens (int): The maximum number of new tokens to generate.
        stream (bool): Whether to stream the output (not implemented in this function).
        enable_thinking (bool): Whether to enable "thinking" mode in the chat template.
        
    Returns:
        str: The generated response from the model.
    """
    
    # Format chat using tokenizer's chat template
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})
        
    # Build the prompt using the chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = enable_thinking
    )

    # Tokenize the prompt and generate a response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Check if sampling parameters are provided
    sampling_params = ['temperature', 'top_p', 'top_k', 'repetition_penalty']
    has_sampling_params = any(param in kwargs for param in sampling_params)
    
    # Set default do_sample based on presence of sampling parameters
    do_sample = kwargs.pop('do_sample', has_sampling_params)
    
    # Disable gradient calculation for generation
    with torch.no_grad():
        # Generate the output tokens
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            do_sample = do_sample,
            pad_token_id = tokenizer.eos_token_id,
            eos_token_id = tokenizer.eos_token_id,
            streamer = TextStreamer(tokenizer, skip_prompt=True) if stream else None,
            **kwargs
        )
        
    # Extract the generated response and decode it
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Return the generated response
    return response
