from config import model_path
from explorer_model.base_explorer_model import BaseExplorerModel
from explorer_model.llama3_explorer_model import Llama3ExplorerModel
from env_adaptors.base_env_adaptor import BaseEnvAdaptor
from env_adaptors.webshop_adaptor import WebshopAdaptor
from env_adaptors.webshop_llama_adaptor import WebshopLlamaAdaptor
from exp_backend.base_exp_backend import BaseExpBackend
from exp_backend.webshop_exp_vanilla_backend import WebshopExpVanillaBackend

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
    elif env_name == "webshop_llama":
        return WebshopLlamaAdaptor(env_name)
    else:
        raise Exception(f"In utils.py load_adaptor(), env_name ({env_name}) is not recognized.")

def load_exp_backend(env_name: str, storage_path: str, depreiciate_exp_store_path: str) -> BaseExpBackend:
    if env_name == "webshop-vanilla":
        return WebshopExpVanillaBackend(env_name, storage_path, depreiciate_exp_store_path)
    else:
        raise Exception(f"In utils.py load_exp_backend(), env_name ({env_name}) is not recognized.")