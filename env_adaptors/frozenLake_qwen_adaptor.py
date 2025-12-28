from .frozenLake_adaptor import FrozenLakeAdaptor
import re

QWEN_FROZENLAKE_SYSTEM_PROMPT = "You are an intelligent exploration agent navigating a frozen lake. Your goal is to reach the destination with highest possible score while avoiding holes. Analyze the current position and decide the next move. Respond with only the action number (0, 1, 2, or 3) without any additional explanation or formatting."


class FrozenLakeQwenAdaptor(FrozenLakeAdaptor):
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
        # Build per-destination guidance: how many steps to move in each direction
        # from current position to reach each destination.
        dest_detail_lines = []
        cur_r, cur_c = cur_pos
        for i, (r, c) in enumerate(destinations, start=1):
            up_steps = max(cur_r - r, 0)
            down_steps = max(r - cur_r, 0)
            left_steps = max(cur_c - c, 0)
            right_steps = max(c - cur_c, 0)
            moves = []
            if up_steps > 0:
                moves.append(f"{up_steps} line up")
            if down_steps > 0:
                moves.append(f"{down_steps} line down")
            if left_steps > 0:
                moves.append(f"{left_steps} column left")
            if right_steps > 0:
                moves.append(f"{right_steps} column right")

            move_str = ", ".join(moves) + " step(s)" if moves else "Already at destination (0 moves)."
            dest_detail_lines.append(
                f"- Destination {i}: ({r}, {c}). It is {move_str} compare to current position ({cur_r}, {cur_c}). "
                f"You do NOT have to go directly; always avoid holes. Explore more and take detours to find a safe way around."
            )
        destinations_detail_str = "\n".join(dest_detail_lines) if dest_detail_lines else "- (none)"
        map_rows = self.env.unwrapped.nrow
        map_cols = self.env.unwrapped.ncol

        user_prompt = f"""
Map Size: {map_rows} rows x {map_cols} columns
Current Position: {cur_pos}

There are a total of {len(destinations)} destination(s).
Destinations: {destinations_str}
For each destination, the steps below describe how many moves are needed in each direction from your CURRENT position:
{destinations_detail_str}
The highest possible score achievable from these destinations is 1.0.

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
                # Extract the score achieved in the next state (default to 0 if not present)
                achieved_score = exp.get('st1', {}).get('score', 0)
                
                summary = exp.get('voyager_summary')
                max_score = exp.get('max_score')
                gen_score = exp.get('generative_score')

                # Add warning/success logic
                tile_warning = ""
                if next_tile == 'H':
                    tile_warning = f" HOLE - AVOID ACTION {action_taken}!"
                    forbidden_actions.append(action_taken)
                elif next_tile == 'G':
                    # Only treat MAX score (>= 1.0) as the true target for BEST CHOICE
                    if achieved_score >= 1.0:
                        tile_warning = f" MAX REWARD GOAL (Score: {achieved_score}) - TAKE ACTION {action_taken}!"
                        goal_actions.append(action_taken)
                    else:
                        # For sub-optimal goals (e.g., 0.5), do NOT add to goal_actions
                        tile_warning = f" SUB-OPTIMAL GOAL (Score: {achieved_score}) - Can you find a better one?"

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
            
            # This triggers only if a 1.0 score goal was found
            if goal_actions:
                user_prompt += f"""!!! BEST CHOICE: Action(s) {goal_actions} will reach the HIGHEST SCORE GOAL directly. Choose this!

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
3. DO NOT simply repeat a "safe" action if it moves you AWAY from the destination. Explore unexplored areas.
4. DO NOT satisfy with low scores (e.g. 0.5). Aim for the HIGHEST SCORE (1.0). Explore unexplored areas to search for best score.
5. AVOID actions strictly forbidden (leading to Holes). Never take actions that lead to holes.

Which action effectively moves you closer to the HIGHEST SCORE goal? Respond with only the action number (0, 1, 2, or 3).
"""

        # Construct the prompt in Qwen ChatML format
        prompt = (
            f"<|im_start|>system\n{QWEN_FROZENLAKE_SYSTEM_PROMPT}\n<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
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