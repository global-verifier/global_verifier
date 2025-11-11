from config import model_path
from explorer_model.base_explorer_model import BaseExplorerModel
from explorer_model.llama3_explorer_model import Llama3ExplorerModel
from env_adaptors.base_env_adaptor import BaseEnvAdaptor
from env_adaptors.webshop_adaptor import WebshopAdaptor
from datetime import datetime

def load_explorer_model(model_name: str) -> BaseExplorerModel:
    if model_name == "llama3":
        return Llama3ExplorerModel(model_path[model_name])
    if model_name == "llama3.1":
        return Llama3ExplorerModel(model_path[model_name])
    # elif model_name == "qwen2.5":
    #     pass
    # elif model_name == "qwen3":
    #     pass
    else:
        raise Exception(f"In utils.py load_model(), model_name ({model_name}) is not recognized.")

def load_adaptor(env_name: str) -> BaseEnvAdaptor:
    if env_name == "webshop":
        return WebshopAdaptor(env_name)
    else:
        raise Exception(f"In utils.py load_adaptor(), env_name ({env_name}) is not recognized.")

def log_flush(fileIO, txt: str):
    """
    Write and flush to the disk.
    """
    fileIO.write(txt)
    fileIO.write("\n")
    fileIO.flush()

def get_timestamp() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
