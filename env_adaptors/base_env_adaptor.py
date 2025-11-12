class BaseEnvAdaptor:
    def __init__(self, env_name):
        self.env_name = env_name

    def initialize_env(self):
        raise NotImplementedError


    def step(self):
        raise NotImplementedError


    def get_action_prompt(self, instruction: str, state: dict, available_actions: list) -> str:
        raise NotImplementedError


    def check_action_valid(self, action, available_actions):
        raise NotImplementedError

