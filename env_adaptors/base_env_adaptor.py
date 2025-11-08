class BaseEnvAdaptor:
    def __init__(self, env_name):
        self.env_name = env_name

    def initialize_env(self):
        pass

    def convert_to_state(self, obs):
        pass
