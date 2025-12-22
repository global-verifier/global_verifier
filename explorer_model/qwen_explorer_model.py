from .base_explorer_model import BaseExplorerModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _extract_qwen_assistant_response(full_text: str) -> str:
    """
    Extract assistant response from generated text.
    Qwen ChatML uses <|im_end|> to terminate a message.
    """
    if "<|im_end|>" in full_text:
        full_text = full_text.split("<|im_end|>", 1)[0]
    return full_text.strip()


class QwenExplorerModel(BaseExplorerModel):
    def __init__(self, model_path: str, max_new_tokens: int = 64):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
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

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
        )
        # Strip the prompt tokens
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        action = _extract_qwen_assistant_response(response)
        return action


