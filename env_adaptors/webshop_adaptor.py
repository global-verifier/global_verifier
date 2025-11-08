import gym
from base_env_adaptor import BaseEnvAdaptor
from env_config import webshop_config

class WebshopAdaptor(BaseEnvAdaptor):
    def __init__(self, env_name):
        super().__init__(env_name)
        self.env = gym.make(webshop_config['id'], observation_mode=webshop_config['observation_mode'], num_products=webshop_config['num_products'])
        
    def initialize_env(self):
        self.env.reset()

    def convert_to_state(self, obs):
        pass
