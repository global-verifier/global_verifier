from .webshop_adaptor import WebshopAdaptor

LLAMA3_WEBSHOP_SYSTEM_PROMPT = "You are an intelligent exploration agent that navigates through environments to accomplish tasks. Your goal is to analyze the current state, understand the task instruction, and determine the next action to take. Respond with only the action you want to execute, without any additional explanation or formatting."

class WebshopLlamaAdaptor(WebshopAdaptor):
    def __init__(self, env_name):
        super().__init__(env_name)

    def get_action_prompt(self, retrieved_experiences=None):
        """生成用于LLM获取下一个action的prompt"""
        if retrieved_experiences is None:
            retrieved_experiences = []

        state = self.get_state()
        instruction = self.get_instruction()
            
        # Construct the user prompt
        action_status = self.get_available_actions()
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

        # Add current episode's action history to avoid repeating ineffective actions
        action_path = self.get_action_path()
        if action_path:
            user_prompt += f"""
Actions you have already taken in this episode: {action_path}
Try not repeat action already in the action path.
Do NOT repeat actions that didn't change the state. If you already clicked an option, try a different one.
"""

        # Add historical experiences if available
        if retrieved_experiences:
            user_prompt += f"""\n--- Historical Experience from Similar States ---
You have visited this state before. Here are {len(retrieved_experiences)} previous experience(s):

"""
            for idx, exp in enumerate(retrieved_experiences, 1):
                action_taken = exp.get('action', 'N/A')
                next_url = exp.get('st1', {}).get('url', 'N/A')
                max_score = exp.get('max_score', None)
                summary = exp.get('voyager_summary')
                gen_score = exp.get('generative_score')
                user_prompt += f"""Experience {idx}:
  Action taken: {action_taken}
  Result URL: {next_url}
  Max score achievable: {max_score if max_score is not None else 'Unknown (may not lead to success)'}

"""
                if summary:
                    user_prompt += f"""  Summary for this step is: {summary}
"""
                if gen_score is not None:
                    user_prompt += f"""  LLM analyzed score for this action is: {gen_score}
"""
            user_prompt += """IMPORTANT: Choose the action with the HIGHEST max_score! 
Actions with max_score=None may not lead to success. Prefer actions with a known positive score.
---

"""

        user_prompt += """You goal is to buy the most suitable product that satisfies the task instruction and get the maximum score (1.0).
Please consider selecitng required options before buying.
Based on the current state and task instruction, what is the next action you should take?

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
