from config import model_path
from explorer_model.base_explorer_model import BaseExplorerModel
from explorer_model.llama3_explorer_model import Llama3ExplorerModel
from explorer_model.qwen_explorer_model import QwenExplorerModel
from env_adaptors.base_env_adaptor import BaseEnvAdaptor
from exp_backend.base_exp_backend import BaseExpBackend


def load_explorer_model(model_name: str) -> BaseExplorerModel:
    if model_name == "llama3-8b":
        return Llama3ExplorerModel(model_path[model_name])
    if model_name == "llama3.1":
        return Llama3ExplorerModel(model_path[model_name])
    if model_name == "llama3.2-3b":
        return Llama3ExplorerModel(model_path[model_name])
    if model_name == "qwen2-7b":
        return QwenExplorerModel(model_path[model_name])
    if model_name == "qwen2.5-7b":
        return QwenExplorerModel(model_path[model_name])
    if model_name == "qwen3-8b":
        return QwenExplorerModel(model_path[model_name])
    if model_name == "qwen3-30b":
        return QwenExplorerModel(model_path[model_name])
    else:
        raise Exception(f"In utils.py load_model(), model_name ({model_name}) is not recognized.")


def load_adaptor(env_name: str, **kwargs) -> BaseEnvAdaptor:
    if env_name == "webshop_llama":
        from env_adaptors.webshop_llama_adaptor import WebshopLlamaAdaptor
        return WebshopLlamaAdaptor(env_name)
    if env_name == "webshop_qwen":
        from env_adaptors.webshop_qwen_adaptor import WebshopQwenAdaptor
        return WebshopQwenAdaptor(env_name)
    elif env_name == "frozenlake_llama":
        from env_adaptors.frozenLake_llama_adaptor import FrozenLakeLlamaAdaptor
        return FrozenLakeLlamaAdaptor(
            env_name,
            desc=kwargs.get("desc"),
            goal_rewards=kwargs.get("goal_rewards"),
        )
    elif env_name == "frozenlake_qwen":
        from env_adaptors.frozenLake_qwen_adaptor import FrozenLakeQwenAdaptor
        return FrozenLakeQwenAdaptor(
            env_name,
            desc=kwargs.get("desc"),
            goal_rewards=kwargs.get("goal_rewards"),
        )
    elif env_name == "cartpole_llama":
        from env_adaptors.cartPole_llama_adaptor import CartPoleLlamaAdaptor
        return CartPoleLlamaAdaptor(env_name)
    elif env_name == "cartpole_qwen":
        from env_adaptors.cartPole_qwen_adaptor import CartPoleQwenAdaptor
        return CartPoleQwenAdaptor(env_name)
    elif env_name == "mountaincar_llama":
        from env_adaptors.mountainCar_llama_adaptor import MountainCarLlamaAdaptor
        return MountainCarLlamaAdaptor(env_name, force=kwargs.get("force"))
    elif env_name == "mountaincar_qwen":
        from env_adaptors.mountainCar_qwen_adaptor import MountainCarQwenAdaptor
        return MountainCarQwenAdaptor(env_name, force=kwargs.get("force"))
    else:
        raise Exception(f"In utils.py load_adaptor(), env_name ({env_name}) is not recognized.")


def load_exp_backend(env_name: str, storage_path: str, depreiciate_exp_store_path: str, explorer_model=None, **kwargs) -> BaseExpBackend:
    # Extract common optional kwargs
    log_dir = kwargs.pop("log_dir", None)
    start_timestep = kwargs.pop("start_timestep", None)
    threshold = kwargs.pop("threshold", None)
    decay_rate = kwargs.pop("decay_rate", None)

    if env_name == "webshop-vanilla":
        from exp_backend.webshop_exp_vanilla_backend import WebshopExpVanillaBackend
        return WebshopExpVanillaBackend(env_name, storage_path, depreiciate_exp_store_path, log_dir=log_dir)
    elif env_name == "frozenlake-vanilla":
        from exp_backend.frozenLake_exp_vanilla_backend import FrozenLakeExpVanillaBackend
        return FrozenLakeExpVanillaBackend(env_name, storage_path, depreiciate_exp_store_path, log_dir=log_dir)
    elif env_name == "cartpole-vanilla":
        from exp_backend.cartPole_exp_vanilla_backend import CartPoleExpVanillaBackend
        return CartPoleExpVanillaBackend(env_name, storage_path, depreiciate_exp_store_path, log_dir=log_dir)
    elif env_name == "mountaincar-vanilla":
        from exp_backend.mountainCar_exp_vanilla_backend import MountainCarExpVanillaBackend
        return MountainCarExpVanillaBackend(env_name, storage_path, depreiciate_exp_store_path, log_dir=log_dir)
    elif env_name == "frozenlake-memorybank":
        from exp_backend.frozenLake_exp_memorybank_backend import FrozenLakeExpMemoryBankBackend
        return FrozenLakeExpMemoryBankBackend(
            env_name,
            storage_path,
            depreiciate_exp_store_path,
            log_dir=log_dir,
            start_timestep=start_timestep,
            threshold=threshold,
            decay_rate=decay_rate,
            **kwargs,
        )
    elif env_name == "mountaincar-memorybank":
        from exp_backend.mountainCar_exp_memorybank_backend import MountainCarExpMemoryBankBackend
        return MountainCarExpMemoryBankBackend(
            env_name,
            storage_path,
            depreiciate_exp_store_path,
            log_dir=log_dir,
            start_timestep=start_timestep,
            threshold=threshold,
            decay_rate=decay_rate,
            **kwargs,
        )
    elif env_name == "webshop-memorybank":
        from exp_backend.webshop_exp_memorybank_backend import WebshopExpMemoryBankBackend
        return WebshopExpMemoryBankBackend(
            env_name,
            storage_path,
            depreiciate_exp_store_path,
            log_dir=log_dir,
            start_timestep=start_timestep,
            threshold=threshold,
            decay_rate=decay_rate,
            **kwargs,
        )
    # Voyager backends (需要 explorer_model 来生成总结)
    elif env_name == "frozenlake-voyager":
        from exp_backend.frozenLake_exp_voyager_backend import FrozenLakeExpVoyagerBackend
        return FrozenLakeExpVoyagerBackend(
            env_name,
            storage_path,
            depreiciate_exp_store_path,
            explorer_model,
            log_dir=log_dir,
            **kwargs,
        )
    elif env_name == "mountaincar-voyager":
        from exp_backend.mountainCar_exp_voyager_backend import MountainCarExpVoyagerBackend
        return MountainCarExpVoyagerBackend(
            env_name,
            storage_path,
            depreiciate_exp_store_path,
            explorer_model,
            log_dir=log_dir,
            **kwargs,
        )
    elif env_name == "webshop-voyager":
        from exp_backend.webshop_exp_voyager_backend import WebshopExpVoyagerBackend
        return WebshopExpVoyagerBackend(
            env_name,
            storage_path,
            depreiciate_exp_store_path,
            explorer_model,
            log_dir=log_dir,
            **kwargs,
        )
    # Generative backends (需要 explorer_model 来检索时打分排序)
    elif env_name == "frozenlake-generative":
        from exp_backend.frozenLake_exp_generative_backend import FrozenLakeExpGenerativeBackend
        return FrozenLakeExpGenerativeBackend(
            env_name,
            storage_path,
            depreiciate_exp_store_path,
            explorer_model,
            log_dir=log_dir,
            **kwargs,
        )
    elif env_name == "mountaincar-generative":
        from exp_backend.mountainCar_exp_generative_backend import MountainCarExpGenerativeBackend
        return MountainCarExpGenerativeBackend(
            env_name,
            storage_path,
            depreiciate_exp_store_path,
            explorer_model,
            log_dir=log_dir,
            **kwargs,
        )
    elif env_name == "webshop-generative":
        from exp_backend.webshop_exp_generative_backend import WebshopExpGenerativeBackend
        return WebshopExpGenerativeBackend(
            env_name,
            storage_path,
            depreiciate_exp_store_path,
            explorer_model,
            log_dir=log_dir,
            **kwargs,
        )
    else:
        raise Exception(f"In utils.py load_exp_backend(), env_name ({env_name}) is not recognized.")
