from .cartPole_adaptor import CartPoleAdaptor
import re

QWEN_CARTPOLE_SYSTEM_PROMPT = """You are an intelligent control agent for the CartPole environment. Your goal is to balance a pole on a moving cart by choosing to push the cart left or right.

CRITICAL RULES:
1. The pole falls if the angle exceeds ±12 degrees
2. The cart fails if it moves beyond ±2.4 units from center
3. Learn from past experiences to avoid known failure patterns
4. The goal is to keep the pole balanced for as long as possible
5. Respond with only the action number (0 or 1) without explanation"""


class CartPoleQwenAdaptor(CartPoleAdaptor):
    def __init__(self, env_name):
        super().__init__(env_name)

    def get_action_prompt(self, retrieved_experiences=None):
        if retrieved_experiences is None:
            retrieved_experiences = []

        # Get current state information
        available_actions = self.get_available_actions()
        state = self.get_state()

        # Build state description
        position_desc = self._get_position_description(state['x_bin'])
        angle_desc = self._get_angle_description(state['theta_bin'])
        velocity_desc = self._get_velocity_description(state['x_dot_sign'], state['theta_dot_sign'])

        user_prompt = f"""
Current Episode Steps: {self.episode_length}
Current Total Reward: {self.episode_reward}

Current State:
  Cart Position: {position_desc} (bin {state['x_bin']})
  Pole Angle: {angle_desc} (bin {state['theta_bin']})
  {velocity_desc}

Available Actions:
  0 = Push cart to the LEFT
  1 = Push cart to the RIGHT

"""

        # Add historical experiences if available
        if retrieved_experiences:
            # Categorize experiences
            failure_actions = []
            success_actions = []

            for exp in retrieved_experiences:
                action = exp.get('action', 'N/A')
                st1 = exp.get('st1', {})
                summary = exp.get('voyager_summary')
                gen_score = exp.get('generative_score')

                # Check if this action led to failure
                # (We can infer failure if the next state is very different or episode ended)
                if self._is_failure_state(st1):
                    failure_actions.append((action, st1, summary, gen_score))
                else:
                    success_actions.append((action, st1, summary, gen_score))

            user_prompt += f"""\n--- Historical Experience from Similar States ---
Based on {len(retrieved_experiences)} previous attempt(s) from similar state:

"""

            if failure_actions:
                user_prompt += f"""DANGEROUS ACTIONS (led to failure):
"""
                for action, st1, summary, gen_score in failure_actions[:3]:  # Show top 3
                    user_prompt += f"""  Action {action} → Failed (pole fell or cart out of bounds)
"""
                    if summary:
                        user_prompt += f"""    Summary for this step is: {summary}
"""
                    if gen_score is not None:
                        user_prompt += f"""    LLM analyzed score for this action is: {gen_score}
"""
                user_prompt += f"""
AVOID: Actions {set(fa[0] for fa in failure_actions)} have high failure rate from this state!

"""

            if success_actions:
                user_prompt += f"""SUCCESSFUL ACTIONS (kept pole balanced):
"""
                for action, st1, summary, gen_score in success_actions[:3]:  # Show top 3
                    result_desc = self._get_position_description(st1.get('x_bin', 0))
                    result_angle = self._get_angle_description(st1.get('theta_bin', 0))
                    user_prompt += f"""  Action {action} → Cart: {result_desc}, Angle: {result_angle}
"""
                    if summary:
                        user_prompt += f"""    Summary for this step is: {summary}
"""
                    if gen_score is not None:
                        user_prompt += f"""    LLM analyzed score for this action is: {gen_score}
"""
                user_prompt += f"""
RECOMMENDED: Actions {set(sa[0] for sa in success_actions)} worked well from similar states.

"""

            user_prompt += """Consider these experiences when deciding your next action.
---

"""

        user_prompt += """Task: Choose the BEST action to keep the pole balanced.
- Prioritize keeping the pole angle near vertical (0°)
- Prevent the cart from reaching the edges (±2.4)
- Use past experiences to avoid known failure patterns
- Consider both position and angle when deciding

Respond with only ONE action number (0 or 1).
"""

        # Construct the prompt in Qwen ChatML format
        prompt = (
            f"<|im_start|>system\n{QWEN_CARTPOLE_SYSTEM_PROMPT}\n<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt

    def _get_position_description(self, x_bin):
        """Convert position bin to human-readable description."""
        descriptions = [
            "FAR LEFT (near edge!)",
            "LEFT edge",
            "Slightly left",
            "CENTER",
            "Slightly right",
            "RIGHT edge",
            "FAR RIGHT (near edge!)",
            "OUT OF BOUNDS"
        ]
        return descriptions[min(x_bin, len(descriptions)-1)]

    def _get_angle_description(self, theta_bin):
        """Convert angle bin to human-readable description."""
        descriptions = [
            "FALLING LEFT (critical!)",
            "Tilted left",
            "Slightly left",
            "VERTICAL (perfect!)",
            "Slightly right",
            "Tilted right",
            "FALLING RIGHT (critical!)",
            "FALLEN"
        ]
        return descriptions[min(theta_bin, len(descriptions)-1)]

    def _get_velocity_description(self, x_dot_sign, theta_dot_sign):
        """Describe movement direction."""
        cart_dir = "moving LEFT" if x_dot_sign < 0 else ("moving RIGHT" if x_dot_sign > 0 else "stationary")
        pole_dir = "falling LEFT" if theta_dot_sign < 0 else ("falling RIGHT" if theta_dot_sign > 0 else "stable")
        return f"Cart: {cart_dir}, Pole: {pole_dir}"

    def _is_failure_state(self, state):
        """Check if a state indicates failure."""
        x_bin = state.get('x_bin', 0)
        theta_bin = state.get('theta_bin', 0)
        # Failure if position or angle is at extreme
        return x_bin == 0 or x_bin >= 6 or theta_bin == 0 or theta_bin >= 6

    def format_action(self, action):
        """
        Format the action from LLM output to valid action integer.
        Extracts exactly one digit (0 or 1) from the response.
        """
        action = action.strip()
        # Find all numbers that are 0 or 1
        matches = re.findall(r'[0-1]', action)

        if len(matches) == 0:
            raise ValueError(f"Could not extract valid action (0 or 1) from: {action}")
        elif len(matches) > 1:
            raise ValueError(f"Multiple actions found ({matches}) in: {action}")
        else:
            return int(matches[0])


