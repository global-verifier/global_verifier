from .base_explorer_model import BaseExplorerModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _extract_qwen_assistant_response(full_text: str) -> str:
    """
    Extract assistant response from generated text.
    Qwen 2.5 uses Llama 3 style special tokens: <|eot_id|>, <|endoftext|>
    Older Qwen versions use ChatML: <|im_end|>
    
    Processing steps:
    1. First remove newlines and normalize whitespace (CRITICAL for special token matching)
    2. Then remove special tokens
    3. Finally clean up remaining whitespace
    """
    # Step 1: Remove newlines first (prevents breaking special tokens like "<|\n|eot_id|>")
    full_text = full_text.replace('\n', '').replace('\r', '').replace('\t', '')
    
    # Step 2: Normalize multiple spaces into single spaces
    full_text = ' '.join(full_text.split())
    
    # Step 3: Remove special tokens
    special_tokens = [
        '<|eot_id|>',
        '<|endoftext|>',
        '<|end_of_text|>',
        '<|im_end|>',  # Older Qwen ChatML format
        '<|end|>',
        '<|assistant|>',
        '<|user|>',
        '<|system|>',
    ]
    
    for token in special_tokens:
        if token in full_text:
            full_text = full_text.split(token, 1)[0]
    
    # Step 4: Clean up any remaining special token fragments
    # Handle partial tokens like "<|" or "|>"
    if '<|' in full_text:
        # Keep only content before any opening special token
        full_text = full_text.split('<|', 1)[0]
    
    # Step 5: Final cleanup
    return full_text.strip()


class QwenExplorerModel(BaseExplorerModel):
    def __init__(self, model_path: str, max_new_tokens: int = 64):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_new_tokens = max_new_tokens

    def digest_goal(self, goal: str) -> None:
        pass

    def get_next_action(self, get_action_prompt: str) -> str:
        """
        Generate next action using Qwen chat template style prompt.
        The prompt should already be in ChatML format ending with <|im_start|>assistant.
        """
        model_inputs = self.tokenizer(
            [get_action_prompt],
            return_tensors="pt"
        ).to(self.model.device)

        # Get special token IDs for early stopping
        eos_token_id = self.tokenizer.eos_token_id
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,  # Use greedy decoding for consistency
            num_beams=1,      # No beam search
            temperature=0.9,
            top_p=0.95,
            # top_k=50,
            # repetition_penalty=1.2,
            # length_penalty=1.0,
            # early_stopping=True,
            # max_time=10.0,
            # max_length=100,
        )
        # Strip the prompt tokens
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        # Decode with skip_special_tokens=True to remove most special tokens
        # Then use our custom function to clean up any remaining ones
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        action = _extract_qwen_assistant_response(response)
        return action


