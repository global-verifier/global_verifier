from config import model_path, api_model_name
from explorer_model.base_explorer_model import BaseExplorerModel
from explorer_model.llama3_explorer_model import Llama3ExplorerModel
from explorer_model.qwen_explorer_model import QwenExplorerModel
from explorer_model.mistral_explorer_model import MistralExplorerModel
from env_adaptors.base_env_adaptor import BaseEnvAdaptor
from exp_backend.base_exp_backend import BaseExpBackend
from explorer_model.internlm_explorer_model import InternLMExplorerModel
from explorer_model.deepseek_explorer_model import DeepSeekExplorerModel
from explorer_model.openai_explorer_model import OpenAIExplorerModel


def load_explorer_model(model_name: str, use_api: bool = False) -> BaseExplorerModel:
    # use api model
    if use_api:
        return OpenAIExplorerModel(api_model_name[model_name])
    # use local model
    if "llama" in model_name:
        return Llama3ExplorerModel(model_path[model_name])
    if "qwen" in model_name:
        return QwenExplorerModel(model_path[model_name])
    if "mistral" in model_name:
        return MistralExplorerModel(model_path[model_name])
    if "internlm" in model_name:
        return InternLMExplorerModel(model_path[model_name])
    if "deepseek" in model_name:
        return DeepSeekExplorerModel(model_path[model_name])
    raise Exception(f"In utils.py load_model(), model_name ({model_name}) is not recognized.")


def load_adaptor(env_name: str, model_name: str, **kwargs) -> BaseEnvAdaptor:
    env_name_raw = env_name
    env_parts = env_name_raw.split("_", 1)
    base_env = env_parts[0]

    if base_env not in ["webshop", "frozenlake", "mountaincar"]:
        raise Exception(f"In utils.py load_adaptor(), env_name ({env_name_raw}) is not recognized, must be one of [webshop, frozenlake, mountaincar].")
    if model_name is None:
        raise Exception(f"In utils.py load_adaptor(), model_name is None and env_name ({env_name_raw}) does not encode a model.")

    # load adaptor
    if base_env == "webshop":
        from env_adaptors.webshop_adaptor import WebshopAdaptor
        return WebshopAdaptor(
            base_env,
            model_name,
            enable_confirm_purchase=kwargs.get("enable_confirm_purchase"),
            correct_index=kwargs.get("correct_index"),
            session=kwargs.get("session"),
        )
    elif base_env == "frozenlake":
        from env_adaptors.frozenLake_adaptor import FrozenLakeAdaptor
        return FrozenLakeAdaptor(
            base_env,
            model_name,
            desc=kwargs.get("desc"),
            goal_rewards=kwargs.get("goal_rewards"),
        )
    elif base_env == "mountaincar":
        from env_adaptors.mountainCar_adaptor import MountainCarAdaptor
        return MountainCarAdaptor(base_env, model_name, force=kwargs.get("force"))
    else:
        raise Exception(f"In utils.py load_adaptor(), env_name ({env_name_raw}) is not recognized.")


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
