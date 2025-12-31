def extract_llama3_assistant_response(full_text):
    assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
    if assistant_marker not in full_text:
        raise ValueError(f"""Assistant marker "{assistant_marker}" not found in the full text: "{assistant_marker}".""")
    response = full_text.split(assistant_marker, 1)[1].strip()

    # Strip common special tokens that models sometimes append to the end.
    # WebShop action parser expects clean actions like: search[...], click[...]
    trailing_tokens = [
        "<|eot_id|>",
        "<|endoftext|>",
        "<|end_of_text|>",
        "<|im_end|>",
        "<|end|>",
    ]
    for tok in trailing_tokens:
        if tok in response:
            response = response.split(tok, 1)[0]

    return response.strip()

def extract_mistral_response(full_text):
    """
    Extracts the response from Mistral/Ministral formatted text.
    Assumes the prompt ends with [/INST].
    """
    marker = "[/INST]"
    if marker not in full_text:
        # Fallback logic: if [/INST] is not found, try to identify if the text is just the response 
        # or raise an error. For safety, we raise error as the prompt structure is expected.
        raise ValueError(f"""Marker "{marker}" not found in the full text.""")
    
    response = full_text.split(marker, 1)[1].strip()

    # Mistral models usually use </s> as EOS
    trailing_tokens = [
        "</s>",
        "<|endoftext|>",
    ]
    for tok in trailing_tokens:
        if tok in response:
            response = response.split(tok, 1)[0]

    remove_tokens = [
        "<s>",
    ]
    # Some models may accidentally emit special tokens inside the content (not only as trailing tokens).
    # Remove them wherever they appear.
    for tok in remove_tokens:
        if tok:
            response = response.replace(tok, "")

    return response.strip()

def extract_internlm_response(full_text):
    """
    提取 InternLM3 的回复。
    """
    assistant_marker = "<|im_start|>assistant\n"
    # 容错处理：有时模型生成或 tokenizer 解码可能不包含换行
    if assistant_marker not in full_text:
        assistant_marker = "<|im_start|>assistant"

    if assistant_marker not in full_text:
        raise ValueError(f"Assistant marker '{assistant_marker.strip()}' not found in the full text.")

    response = full_text.split(assistant_marker, 1)[1].strip()

    # 移除常见的结束符
    trailing_tokens = [
        "<|im_end|>",
        "<|endoftext|>",
    ]
    for tok in trailing_tokens:
        if tok in response:
            response = response.split(tok, 1)[0]

    return response.strip()

def extract_deepseek_response(full_text):
    """
    提取 DeepSeek-Coder-V2 的回复。
    """
    # 按照 format_full_deepseek_prompt 的构造，回复紧跟在 "Assistant: " 之后
    # 为了防止 User 输入中包含 Assistant:，建议使用更精确的切分，或者 split 最后一个。
    # 这里使用我们在 Prompt 中构造的 "\n\nAssistant: "
    marker = "\n\nAssistant: "
    if marker not in full_text:
        marker = "Assistant: " # 降级尝试

    if marker not in full_text:
        raise ValueError(f"Marker 'Assistant:' not found in the full text.")

    # 提取 marker 之后的内容
    response = full_text.split(marker, 1)[1].strip()

    # DeepSeek Coder V2 的特殊结束符
    trailing_tokens = [
        "<｜end▁of▁sentence｜>", 
        "<|endoftext|>",
    ]
    for tok in trailing_tokens:
        if tok in response:
            response = response.split(tok, 1)[0]

    return response.strip()
