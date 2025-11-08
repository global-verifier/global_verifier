from utils import load_explorer_model, load_adaptor
from config import explorer_settings

class Explorer:
    def __init__(self, model_name: str, env_name: str):
        self.explorer_model = load_explorer_model(model_name)
        self.max_steps = explorer_settings["max_steps"]
        self.adaptor = load_adaptor(env_name)
        self.goal = None


    def explore(self):
        # Digest the goal
        # rest the status
        self.adaptor.initialize_env()
        for i in range(self.max_steps):
