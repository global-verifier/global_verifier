from .frozenLake_adaptor import FrozenLakeAdaptor
import re

from .adaptor_prompt_factory import build_frozenlake_user_prompt, FROZENLAKE_SYSTEM_PROMPT

class FrozenLakeLlamaAdaptor(FrozenLakeAdaptor):
    def __init__(self, env_name, desc=None, goal_rewards=None):
        super().__init__(env_name, desc=desc, goal_rewards=goal_rewards)

    def get_action_prompt(self, retrieved_experiences=None):
        if retrieved_experiences is None:
            retrieved_experiences = []
        state = self.get_state()
        user_prompt = build_frozenlake_user_prompt(
            state=state,
            available_actions=self.get_available_actions(),
            destinations=self.destinations,
            goal_rewards=self.goal_rewards,
            map_rows=self.env.unwrapped.nrow,
            map_cols=self.env.unwrapped.ncol,
            retrieved_experiences=retrieved_experiences,
        )
        
        # Construct the prompt in Llama3 format
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{FROZENLAKE_SYSTEM_PROMPT} 
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_prompt}
<|eot_id|> 
<|start_header_id|>assistant<|end_header_id|>
"""
        return prompt

