from .mountainCar_adaptor import MountainCarAdaptor
import re

QWEN_MOUNTAINCAR_SYSTEM_PROMPT = """You are an intelligent control agent for the Mountain Car environment.

ENVIRONMENT:
- The car is in a valley between two hills
- Position range: -1.2 (left edge) to 0.6 (right edge)
- Valley bottom (lowest point): position -0.5
- Goal: reach position 0.5 (top of right hill)
- The car starts near the valley bottom

PHYSICS HINT:
- The car may need to build momentum before climbing uphill
- Going uphill slows the car down, going downhill speeds it up

RULES:
1. Learn from past experiences to find the optimal strategy
2. Respond with only the action number (0, 1, or 2) without explanation"""


class MountainCarQwenAdaptor(MountainCarAdaptor):
    def __init__(self, env_name, force=None):
        super().__init__(env_name, force=force)

    def get_action_prompt(self, retrieved_experiences=None):
        if retrieved_experiences is None:
            retrieved_experiences = []

        # Get current state information
        available_actions = self.get_available_actions()
        state = self.get_state()

        # Build state description using numerical values
        position = state['position']
        velocity = state['velocity']
        position_desc = self._get_position_description(position)
        velocity_desc = self._get_velocity_description(velocity)
        progress_desc = self._get_progress_description(position)

        user_prompt = f"""
Current Episode Steps: {self.episode_length}
Current Total Reward: {self.episode_reward}

Current State:
  Position: {position_desc} (value: {position:.3f})
  Velocity: {velocity_desc} (value: {velocity:.4f})
  Progress: {progress_desc}

Available Actions:
  0 = Push LEFT (accelerate towards left hill)
  1 = NO PUSH (coast with current momentum)
  2 = Push RIGHT (accelerate towards right hill/goal)

"""

        # Add historical experiences if available
        if retrieved_experiences:
            user_prompt += f"""\n--- Historical Experience from This Exact State ---
You have been at this state before. Here are {len(retrieved_experiences)} previous experience(s):

"""
            for idx, exp in enumerate(retrieved_experiences, 1):
                action = exp.get('action', 'N/A')
                st1 = exp.get('st1', {})
                reachable = exp.get('reachable', None)
                path_length = exp.get('path_length', None)
                summary = exp.get('voyager_summary')
                gen_score = exp.get('generative_score')

                user_prompt += f"""Experience {idx}:
  Action taken: {action}
  Result state: pos={st1.get('position', 0):.3f}, vel={st1.get('velocity', 0):.4f}
  Can reach goal: {reachable if reachable is not None else 'Unknown'}
  Steps to goal: {path_length if path_length is not None else 'Unknown'}

"""
                if summary:
                    user_prompt += f"""  Summary for this step is: {summary}
"""
                if gen_score is not None:
                    user_prompt += f"""  LLM analyzed score for this action is: {gen_score}
"""
            # Check if there's a reachable action
            reachable_exps = [e for e in retrieved_experiences if e.get('reachable', False)]
            if reachable_exps:
                best_action = reachable_exps[0]['action']
                user_prompt += f"""If an action can reach the goal, you MUST choose that action.
Action {best_action} can reach the goal. You MUST choose {best_action}.

"""
            else:
                user_prompt += """Choose the action that may lead to the goal.

"""

        user_prompt += """Respond with only ONE action number (0, 1, or 2).
"""

        # Construct the prompt in Qwen ChatML format
        prompt = (
            f"<|im_start|>system\n{QWEN_MOUNTAINCAR_SYSTEM_PROMPT}\n<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt

    def _get_position_description(self, position):
        """Convert position value to human-readable description."""
        # Position ranges from -1.2 (far left) to 0.6 (beyond goal)
        # Goal is at 0.5
        if position >= 0.5:
            desc = "AT GOAL (flag reached!)"
        elif position >= 0.45:
            desc = "NEAR GOAL (almost there!)"
        elif position >= 0.3:
            desc = "RIGHT hill (upper slope)"
        elif position >= 0.1:
            desc = "RIGHT hill (middle slope)"
        elif position >= -0.1:
            desc = "RIGHT hill (lower slope)"
        elif position >= -0.3:
            desc = "VALLEY (right side)"
        elif position >= -0.5:
            desc = "VALLEY bottom (deepest point)"
        elif position >= -0.7:
            desc = "LEFT hill (middle)"
        elif position >= -0.9:
            desc = "LEFT hill (lower)"
        else:
            desc = "FAR LEFT (left edge)"

        # Add distance to goal info
        distance_to_goal = 0.5 - position
        if distance_to_goal > 0:
            desc += f" [{distance_to_goal:.2f} to goal]"
        else:
            desc += " [GOAL REACHED!]"

        return desc

    def _get_velocity_description(self, velocity):
        """Convert velocity value to human-readable description."""
        if velocity < -0.04:
            return "Moving LEFT (fast)"
        elif velocity < -0.01:
            return "Moving LEFT (moderate)"
        elif velocity < -0.002:
            return "Moving LEFT (slow)"
        elif velocity <= 0.002:
            return "STATIONARY (no momentum)"
        elif velocity < 0.01:
            return "Moving RIGHT (slow)"
        elif velocity < 0.04:
            return "Moving RIGHT (moderate)"
        else:
            return "Moving RIGHT (fast)"

    def _get_progress_description(self, position):
        """Describe overall progress towards goal."""
        if position >= 0.5:
            return "SUCCESS - Goal reached"
        elif position >= 0.4:
            return "Very close - One more push"
        elif position >= 0.3:
            return "Making good progress"
        elif position >= 0:
            return "Right side of valley"
        elif position >= -0.5:
            return "Left side of valley"
        else:
            return "On left hill"

    def _is_good_transition(self, pos_before, pos_after, vel_before, vel_after):
        """
        Check if a state transition represents good progress.
        Good progress = moving towards goal or building useful momentum.
        """
        # Reached goal
        if pos_after >= 0.5:
            return True

        # Made progress towards goal (moved right)
        if pos_after > pos_before and pos_after > -0.3:
            return True

        # Building momentum on left side (good strategy)
        if pos_before < -0.5 and abs(vel_after) > abs(vel_before):
            return True

        # Strong rightward velocity
        if vel_after > 0.03:
            return True

        return False

    def format_action(self, action):
        """
        Format the action from LLM output to valid action integer.
        Extracts exactly one digit (0, 1, or 2) from the response.
        """
        action = action.strip()
        # Find all numbers that are 0, 1, or 2
        matches = re.findall(r'[0-2]', action)

        if len(matches) == 0:
            raise ValueError(f"Could not extract valid action (0, 1, or 2) from: {action}")
        elif len(matches) > 1:
            raise ValueError(f"Multiple actions found ({matches}) in: {action}")
        else:
            return int(matches[0])


