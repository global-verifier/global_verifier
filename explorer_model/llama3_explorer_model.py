from base_explorer_model import BaseExplorerModel
from transformers import AutoModelForCausalLM, AutoTokenizer


LLAMA3_WEBSHOP_SYSTEM_PROMPT = "You are an intelligent exploration agent that navigates through environments to accomplish tasks. Your goal is to analyze the current state, understand the task instruction, and determine the next action to take. Respond with only the action you want to execute, without any additional explanation or formatting."

class Llama3ExplorerModel(BaseExplorerModel):
    def __init__(self, model_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def digest_goal(self, goal: str) -> None:
        pass
    
    def get_next_action(self, instruction: str, state: dict, action: str) -> str:
        # TODO: generalize it
        # Format state information
        state_info = ""
        if state:
            if 'url' in state:
                state_info += f"Current URL: {state['url']}\n"
            if 'html' in state:
                state_info += f"Current Page Content:\n{state['html']}\n"
        
        # Format action context
        action_context = ""
        if action is not None:
            if isinstance(action, list) and len(action) > 0:
                action_context = f"Available Actions: {', '.join(map(str, action))}\n"
            elif isinstance(action, str) and action.strip():
                action_context = f"Action Context: {action}\n"
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{LLAMA3_WEBSHOP_SYSTEM_PROMPT} 
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Task Instruction: {instruction}

{state_info}{action_context}

Based on the current state and task instruction, what is the next action you should take?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
        return prompt
        