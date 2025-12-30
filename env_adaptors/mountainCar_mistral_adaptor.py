from .mountainCar_adaptor import MountainCarAdaptor
from .adaptor_prompt_factory import build_mountaincar_user_prompt, MOUNTAINCAR_SYSTEM_PROMPT

class MountainCarMistralAdaptor(MountainCarAdaptor):
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
        
        # Construct the prompt in Ministral / Mistral format
        prompt = f"<s>[INST] {MOUNTAINCAR_SYSTEM_PROMPT}\n\n{user_prompt} [/INST]"
        return prompt