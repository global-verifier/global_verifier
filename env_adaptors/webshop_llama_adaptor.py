import json
import urllib.parse

from .webshop_adaptor import WebshopAdaptor

LLAMA3_WEBSHOP_SYSTEM_PROMPT = "You are an intelligent exploration agent that navigates through environments to accomplish tasks. Your goal is to analyze the current state, understand the task instruction, and determine the next action to take. Respond with only the action you want to execute, without any additional explanation or formatting."

class WebshopLlamaAdaptor(WebshopAdaptor):
    def __init__(self, env_name, enable_confirm_purchase=None, session=None):
        super().__init__(
            env_name,
            enable_confirm_purchase=enable_confirm_purchase,
            session=session,
        )

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
            if "buy now" in action_status["clickables"]:
                user_prompt += """If you think you have meet the requirement, you can click the 'buy now' button to buy the product."""
            user_prompt += """You can only click one button at a time."""

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
"""
                if max_score is None or max_score == 0:
                    # user_prompt += f"""  Result: Score 0.0 (Failed). \n"""
                    user_prompt += f""" \n"""
                elif max_score == 1:
                    user_prompt += f"""  Result: Score 1.0 (PERFECT).
  Max score this action can achieve is 1.0. You SHOULD take the same action: {action_taken}.

"""
                else:
                    user_prompt += f"""  Result: Score {max_score} (SUBOPTIMAL).
  Max score is {max_score}, which is NOT 1.0. This means '{action_taken}' is NOT the best choice (e.g. wrong color or size).
  Do NOT repeat '{action_taken}'. You MUST select a different option to aim for score 1.0!

"""
                if summary:
                    user_prompt += f"""  Summary for this step is: {summary}
"""
                if gen_score is not None:
                    user_prompt += f"""  LLM analyzed score for this action is: {gen_score}
"""
            if "confirm_purchase" in str(state.get("url", "")):
                blacklisted_actions = []
                for exp in retrieved_experiences:
                    max_score = exp.get("max_score", None)
                    action = exp.get("action", None)
                    if max_score in (0, 0.0) and isinstance(action, str) and action.startswith("click["):
                        if action not in blacklisted_actions:
                            blacklisted_actions.append(action)
                if blacklisted_actions:
                    user_prompt += """IMPORTANT: BLACKLISTED ACTIONS
The following actions have been confirmed to result in a score of 0.0 in this specific state. You MUST NOT select them:

"""
                    for action in blacklisted_actions:
                        user_prompt += f"""{action} (Score: 0.0) Please choose a different action from the remaining Clickables to explore potential success.
"""
                    user_prompt += "\n"

        # 对 item_page 做选项黑名单，避免重复选择已选项
        if "item_page" in str(state.get("url", "")):
            selected_option_blacklist = []
            selected_part = str(state.get("url", "")).rsplit("/", 1)[-1]
            decoded_part = urllib.parse.unquote(selected_part)
            selected_dict = json.loads(decoded_part)
            assert isinstance(selected_dict, dict), "selected_dict is not a dict"
            for value in selected_dict.values():
                action = f"click[{str(value).lower()}]"
                if action not in selected_option_blacklist:
                    selected_option_blacklist.append(action)
            if selected_option_blacklist:
                user_prompt += """IMPORTANT: CURRENTLY SELECTED OPTIONS
The following options are already selected on this item page. Do NOT click them again; pick a different option to progress:

"""
                for action in selected_option_blacklist:
                    user_prompt += f"""{action} (already selected). Choose another option from Clickables.
"""
                user_prompt += "\n"

        # 增强结尾的指令
        user_prompt += """IMPORTANT STRATEGY:
1. SCORE 1.0 IS THE ONLY GOAL.
2. If a previous action got 1.0 -> REPEAT IT.
3. If a previous action got anything less than 1.0 (e.g. 0.5, 0.75) -> IT IS WRONG. DO NOT REPEAT IT. CHOOSE A DIFFERENT OPTION.
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
