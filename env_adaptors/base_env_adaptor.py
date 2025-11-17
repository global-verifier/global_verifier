class BaseEnvAdaptor:
    def __init__(self, env_name):
        self.env_name = env_name

    # Consultants
    def check_action_valid(self, action, available_actions):
        raise NotImplementedError

    # Modifiers
    def initialize_env(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def get_action_prompt(self, instruction: str, state: dict, available_actions: list) -> str:
        raise NotImplementedError

    def reconstruct_state(self, exp):
        raise NotImplementedError

    def is_same_state(self, state1, state2):
        raise NotImplementedError
