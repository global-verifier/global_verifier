from .mountainCar_adaptor import MountainCarAdaptor
import re

from .adaptor_prompt_factory import (
    MOUNTAINCAR_SYSTEM_PROMPT,
    build_mountaincar_user_prompt,
)

class MountainCarLlamaAdaptor(MountainCarAdaptor):
    def __init__(self, env_name, force=None):
        super().__init__(env_name, force=force)

    def get_action_prompt(self, retrieved_experiences=None):
        if retrieved_experiences is None:
            retrieved_experiences = []
            
        # Get current state information
        state = self.get_state()
        user_prompt = build_mountaincar_user_prompt(
            episode_length=self.episode_length,
            episode_reward=self.episode_reward,
            state=state,
            retrieved_experiences=retrieved_experiences,
        )
        
        # Construct the prompt in Llama3 format
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{MOUNTAINCAR_SYSTEM_PROMPT} 
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_prompt}
<|eot_id|> 
<|start_header_id|>assistant<|end_header_id|>
"""
        return prompt
    


