import sys
import os
import gym
from .base_env_adaptor import BaseEnvAdaptor
from .env_config import frozenlake_config

class FrozenLakeAdaptor(BaseEnvAdaptor):
    def __init__(self, env_name):
        super().__init__(env_name)
        seed = frozenlake_config.get('random_seed')