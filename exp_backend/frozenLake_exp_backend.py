from .base_exp_backend import BaseExpBackend

class FrozenLakeExpBackend(BaseExpBackend):
    def __init__(self, env_name, storage_path, depreiciate_exp_store_path, log_dir=None):
        # Must define expected_fields BEFORE calling super().__init__()
        # because parent's __init__ calls _is_valid_exp_store() which uses this field
        self.expected_fields = {
            "id": str,
            "action_path": list,
            "st": dict,
            "action": int,
            "st1": dict,
        }
        super().__init__(env_name, storage_path, depreiciate_exp_store_path, log_dir=log_dir)
        