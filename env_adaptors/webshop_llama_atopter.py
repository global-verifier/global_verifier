from webshop_adaptor import WebshopAdaptor

LLAMA3_WEBSHOP_SYSTEM_PROMPT = "You are an intelligent exploration agent that navigates through environments to accomplish tasks. Your goal is to analyze the current state, understand the task instruction, and determine the next action to take. Respond with only the action you want to execute, without any additional explanation or formatting."

class WebshopLlamaAtopter(WebshopAdaptor):
    def __init__(self, env_name):
        super().__init__(env_name)

    def get_next_action_prompt(self, instruction, state, action_status):
        """生成用于LLM获取下一个action的prompt"""
        # Construct the user prompt
        is_search = action_status["has_search_bar"]
        if is_search:
            available_actions = "[search]"
        else:
            available_actions = "[click]"
        
        user_prompt = f"""Task Instruction: {instruction}

Current URL: {state['url']}

Current Webpage Display Text: {state['html']}

Available Actions: {available_actions}

"""
        if not is_search:
            user_prompt += f"""Clickables: {action_status['clickables']}\n"""

        user_prompt += """Based on the current state and task instruction, what is the next action you should take?

If you want to search, response with "search[<search_query>]"
If you want to click, response follow the format: "click[name of the clickable]"
"""
        # Construct the prompt
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{LLAMA3_WEBSHOP_SYSTEM_PROMPT} 
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_prompt}
<|eot_id|> 
<|start_header_id|>assistant<|end_header_id|>
"""
        return prompt
