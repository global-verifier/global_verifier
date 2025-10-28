from config import model_path
from explorer_model.base_explorer_model import BaseExplorerModel
from explorer_model.llama3_explorer_model import Llama3ExplorerModel

def load_explorer_model(model_name: str) -> BaseExplorerModel:
    if model_name == "llama3":
        return Llama3ExplorerModel(model_path[model_name])
    elif model_name == "qwen2.5":
        pass
    elif model_name == "qwen3":
        pass
    else:
        raise Exception(f"In utils.py load_model(), model_name ({model_name}) is not recognized.")