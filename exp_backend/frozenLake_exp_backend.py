from .base_exp_backend import BaseExpBackend

class FrozenLakeExpBackend(BaseExpBackend):
    def __init__(self, env_name, storage_path, depreiciate_exp_store_path):
        super().__init__(env_name, storage_path, depreiciate_exp_store_path)
        