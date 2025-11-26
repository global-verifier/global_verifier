from .frozenLake_adaptor import FrozenLakeAdaptor
import re

LLAMA3_FROZENLAKE_SYSTEM_PROMPT = "You are an intelligent exploration agent navigating a frozen lake. Your goal is to reach the destination while avoiding holes. Analyze the current position and decide the next move. Respond with only the action number (0, 1, 2, or 3) without any additional explanation or formatting."

class FrozenLakeLlamaAdaptor(FrozenLakeAdaptor):
    def __init__(self, env_name):
        super().__init__(env_name)

    def get_action_prompt(self, retrieved_experiences=None):
        if retrieved_experiences is None:
            retrieved_experiences = []
            
        # Construct the user prompt
        available_actions = self.get_available_actions()
        state = self.get_state()
        cur_pos = state['cur_pos']
        tile_type = state['tile_type']
        destination = self.destination
        map_rows = self.env.unwrapped.nrow
        map_cols = self.env.unwrapped.ncol
        
        user_prompt = f"""
Map Size: {map_rows} rows x {map_cols} columns
Destination: {destination}
Current Position: {cur_pos}
Current Tile Type: {tile_type} (S=Start, F=Frozen, H=Hole, G=Goal)

Available Actions: {available_actions}
  0 = Move Left
  1 = Move Down
  2 = Move Right
  3 = Move Up

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
                
                # Add warning for holes
                tile_warning = ""
                if next_tile == 'H':
                    tile_warning = " HOLE - AVOID THIS ACTION!"
                    forbidden_actions.append(action_taken)
                elif next_tile == 'G':
                    tile_warning = " GOAL - TAKE THIS ACTION!"
                    goal_actions.append(action_taken)
                
                user_prompt += f"""  Action taken: {action_taken}
  Result Position: {next_pos}
  Result Tile: {next_tile}{tile_warning}

"""
            
            # Add explicit warning section
            if forbidden_actions or goal_actions:
                user_prompt += "\n"
            if goal_actions:
                    user_prompt += f"""!!! BEST CHOICE: Action(s) {goal_actions} will reach the GOAL directly. Choose this!

"""
                if forbidden_actions:
                    user_prompt += f"""!!! CRITICAL WARNING: Action(s) {forbidden_actions} lead to HOLES and will END THE GAME.
YOU MUST NOT choose {forbidden_actions} from this position!

"""
            
            user_prompt += """Consider these experiences when deciding your next action.
---

"""

        user_prompt += """Based on the current position, task instruction, and past experiences, what is the next action you should take?

REMEMBER: If certain actions lead to holes, you MUST avoid them. Choose the safest action that moves toward the destination.

Respond with only the action number (0, 1, 2, or 3).
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
