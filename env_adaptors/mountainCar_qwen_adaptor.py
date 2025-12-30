from .mountainCar_adaptor import MountainCarAdaptor
import re

from .adaptor_prompt_factory import (
    MOUNTAINCAR_SYSTEM_PROMPT,
    build_mountaincar_user_prompt,
)


class MountainCarQwenAdaptor(MountainCarAdaptor):
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

        # Construct the prompt in Qwen ChatML format
        prompt = (
            f"<|im_start|>system\n{MOUNTAINCAR_SYSTEM_PROMPT}\n<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt




