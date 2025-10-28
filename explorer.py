from utils import load_explorer_model
from config import explorer_settings

class Explorer:
    def __init__(self, model_name: str):
        self.explorer_model = load_explorer_model(model_name)
        self.max_steps = explorer_settings["max_steps"]


    def explore(self, prompt: str):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(input_ids, max_length=100)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)