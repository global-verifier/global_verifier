from base_explorer_model import BaseExplorerModel
from transformers import AutoModelForCausalLM, AutoTokenizer

class Llama3ExplorerModel(BaseExplorerModel):
    def __init__(self, model_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def digest_goal(self, goal: str) -> None:
        pass