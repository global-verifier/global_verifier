class BaseEnvAdaptor:
    def __init__(self, env_name):
        self.env_name = env_name

    def initialize_env(self):
        pass

    def step(self):
        pass

    def get_next_action_prompt(self, instruction: str, state: dict, available_actions: list) -> str:
        pass

    def check_action_valid(self, action, available_actions):
        pass
