from .base_explorer_model import BaseExplorerModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from .model_util import extract_llama3_assistant_response
import torch


LLAMA3_WEBSHOP_SYSTEM_PROMPT = "You are an intelligent exploration agent that navigates through environments to accomplish tasks. Your goal is to analyze the current state, understand the task instruction, and determine the next action to take. Respond with only the action you want to execute, without any additional explanation or formatting."

class Llama3ExplorerModel(BaseExplorerModel):
    def __init__(self, model_path: str, max_new_tokens: int = 64):
        # NOTE:
        # - Using pipeline() will default to moving the entire model onto a single CUDA device,
        #   which easily OOMs for large models (e.g., 70B). We instead load with device_map="auto"
        #   (accelerate) so the model can be sharded across all *visible* GPUs.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Some Llama checkpoints do not define a pad token; fall back to EOS.
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.max_new_tokens = max_new_tokens
        

    def digest_goal(self, goal: str) -> None:
        pass
    
    def get_next_action(self, get_action_prompt: str) -> str:
        model_inputs = self.tokenizer(
            [get_action_prompt],
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                num_beams=1,
            )

        # Strip the prompt tokens
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Keep the same behavior as pipeline(): generated_text = prompt + completion.
        full_text = get_action_prompt + response
        action = extract_llama3_assistant_response(full_text)
        return action
        