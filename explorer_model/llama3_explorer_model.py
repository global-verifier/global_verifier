from .base_explorer_model import BaseExplorerModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from .model_util import extract_llama3_assistant_response


LLAMA3_WEBSHOP_SYSTEM_PROMPT = "You are an intelligent exploration agent that navigates through environments to accomplish tasks. Your goal is to analyze the current state, understand the task instruction, and determine the next action to take. Respond with only the action you want to execute, without any additional explanation or formatting."

class Llama3ExplorerModel(BaseExplorerModel):
    def __init__(self, model_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        

    def digest_goal(self, goal: str) -> None:
        pass
    
    def get_next_action(self, get_action_prompt: str) -> str:
        outputs = self.pipe(get_action_prompt, max_new_tokens=64)
        action = extract_llama3_assistant_response(outputs[0]['generated_text'])
        return action
        