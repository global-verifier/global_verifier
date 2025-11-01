from base_env_adaptor import BaseEnvAdaptor

class WebshopAdaptor(BaseEnvAdaptor):
    def __init__(self, env_name: str):
        super().__init__(env_name)
        
    def convert_to_state(self, obs):
        pass
