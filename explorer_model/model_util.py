
def extract_llama3_assistant_response(full_text):
    assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
    if assistant_marker not in full_text:
        raise ValueError(f"""Assistant marker "{assistant_marker}" not found in the full text: "{assistant_marker}".""")
    response = full_text.split(assistant_marker)[1].strip()
    return response