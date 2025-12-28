from .frozenLake_adaptor import FrozenLakeAdaptor
import re

LLAMA3_FROZENLAKE_SYSTEM_PROMPT = "You are an intelligent exploration agent navigating a frozen lake. Your goal is to reach the destination with highest possible score while avoiding holes. Analyze the current position and decide the next move. Respond with only the action number (0, 1, 2, or 3) without any additional explanation or formatting."

class FrozenLakeLlamaAdaptor(FrozenLakeAdaptor):
    def __init__(self, env_name, desc=None, goal_rewards=None):
        super().__init__(env_name, desc=desc, goal_rewards=goal_rewards)

    def get_action_prompt(self, retrieved_experiences=None):
        if retrieved_experiences is None:
            retrieved_experiences = []
            
        # Construct the user prompt
        available_actions = self.get_available_actions()
        state = self.get_state()
        cur_pos = state['cur_pos']
        tile_type = state['tile_type']
        destinations = self.destinations
        destinations_str = ", ".join([f"({r}, {c})" for r, c in destinations])
        map_rows = self.env.unwrapped.nrow
        map_cols = self.env.unwrapped.ncol
        
        user_prompt = f"""
Map Size: {map_rows} rows x {map_cols} columns
Destinations are: {destinations_str}
Highest Score: 1
Current Position: {cur_pos}
Current Tile Type: {tile_type} (S=Start, F=Frozen, H=Hole, G=Goal)

Available Actions: {available_actions}
  0 = Move Left (Column - 1)
  1 = Move Down (Row + 1)
  2 = Move Right (Column + 1)
  3 = Move Up (Row - 1)

"""
        
        # Add historical experiences if available
        if retrieved_experiences:
            user_prompt += f"""\n--- Historical Experience from Similar States ---
You have been at this position before. Here are {len(retrieved_experiences)} previous experience(s):

"""
            # Collect dangerous and successful actions
            forbidden_actions = []
            goal_actions = []
            
            for exp in retrieved_experiences:
                action_taken = exp.get('action', 'N/A')
                next_pos = exp.get('st1', {}).get('cur_pos', 'N/A')
                next_tile = exp.get('st1', {}).get('tile_type', 'N/A')
                summary = exp.get('voyager_summary')
                max_score = exp.get('max_score')
                gen_score = exp.get('generative_score')
                
                # Add warning for holes
                tile_warning = ""
                if next_tile == 'H':
                    tile_warning = f" HOLE - do not choose action: {action_taken}!"
                    forbidden_actions.append(action_taken)
                elif next_tile == 'G':
                    tile_warning = f" GOAL - TAKE ACTION {action_taken}!"
                    goal_actions.append(action_taken)
                
                user_prompt += f"""  Action taken: {action_taken}
  Result Position: {next_pos}
  Result Tile: {next_tile}{tile_warning}
"""
                if max_score is not None and max_score > 0:
                    user_prompt += f"""  Max score achievable from this path: {max_score}
"""
                if summary:
                    user_prompt += f"""  Summary for this step is: {summary}
"""
                if gen_score is not None:
                    user_prompt += f"""  LLM analyzed score for this action is: {gen_score}
"""
                user_prompt += "\n"
            
            # Add explicit warning section
            if forbidden_actions or goal_actions:
                user_prompt += "\n"
            if goal_actions:
                user_prompt += f"""!!! BEST CHOICE: Action(s) {goal_actions} will reach the GOAL directly. Choose this!

"""
            if forbidden_actions:
                forbidden_list = ", ".join(map(str, dict.fromkeys(forbidden_actions)))
                user_prompt += f"""!!! CRITICAL WARNING: Action(s) {forbidden_list} lead to HOLES and will END THE GAME.
YOU MUST NOT choose {forbidden_list} from this position!

"""
            
            user_prompt += """Consider these experiences when deciding your next action.
---

"""

        user_prompt += """Based on the current position and the destination coordinates:
1. CALCULATE the difference between Current Position and Destinations (Row difference and Column difference).
2. IDENTIFY which action reduces this difference (e.g., if Goal Row > Current Row, you need to increase Row).
3. AVOID actions strictly forbidden (leading to Holes).
4. DO NOT simply repeat a "safe" action if it moves you AWAY from the destination. "Safe" does not mean "Correct".

Which action effectively reduces the distance to the goal? Respond with only the action number (0, 1, 2, or 3).
"""
        
        # Construct the prompt in Llama3 format
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{LLAMA3_FROZENLAKE_SYSTEM_PROMPT} 
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_prompt}
<|eot_id|> 
<|start_header_id|>assistant<|end_header_id|>
"""
        return prompt

    def format_action(self, action):
        """
        Format the action from LLM output to valid action integer.
        Extracts exactly one digit (0-3) from the response.
        Raises error if zero or multiple valid actions found.
        """
        action = action.strip()
        # Find all numbers in the range 0-3
        matches = re.findall(r'[0-3]', action)
        
        if len(matches) == 0:
            raise ValueError(f"Could not extract valid action (0-3) from: {action}")
        elif len(matches) > 1:
            raise ValueError(f"Multiple actions found ({matches}) in: {action}")
        else:
            return int(matches[0])
