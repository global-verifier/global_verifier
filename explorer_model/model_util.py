
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