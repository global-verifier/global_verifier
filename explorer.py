from utils import load_explorer_model, load_adaptor
from config import explorer_settings

class Explorer:
    def __init__(self, model_name: str, env_name: str):
        self.explorer_model = load_explorer_model(model_name)
        self.max_steps = explorer_settings["max_steps"]
        self.adaptor = load_adaptor(env_name)
        self.goal = None


    def explore(self, goal: str):
        # Digest the goal
        self.goal = goal
        # rest the status
        # input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        # output = self.model.generate(input_ids, max_length=100)
        # return self.tokenizer.decode(output[0], skip_special_tokens=True)
        for i in range(self.max_steps):
