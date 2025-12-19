"""
Backend Factory

Factory functions to create memory-enhanced backends for different
environments and memory strategies.

This provides a simple interface to instantiate the appropriate backend
without needing to know the specific class names.
"""

from typing import Optional, Callable

# Import all backend implementations
from .cartPole_memory_backends import (
    CartPoleMemoryBankBackend,
    CartPoleGenerativeBackend,
    CartPoleVoyagerBackend
)
from .mountainCar_memory_backends import (
    MountainCarMemoryBankBackend,
    MountainCarGenerativeBackend,
    MountainCarVoyagerBackend
)
from .webshop_memory_backends import (
    WebshopMemoryBankBackend,
    WebshopGenerativeBackend,
    WebshopVoyagerBackend
)
from .frozenLake_memory_backends import (
    FrozenLakeMemoryBankBackend,
    FrozenLakeGenerativeBackend,
    FrozenLakeVoyagerBackend
)

# Original backends
from .cartPole_exp_backend import CartPoleExpBackend
from .mountainCar_exp_backend import MountainCarExpBackend
from .webshop_exp_backend import WebshopExpBackend
from .frozenLake_exp_backend import FrozenLakeExpBackend


def create_backend(
    env_name: str,
    memory_strategy: str = "none",
    storage_path: Optional[str] = None,
    depreiciate_exp_store_path: Optional[str] = None,
    embedding_func: Optional[Callable] = None,
    memory_persist_dir: Optional[str] = None,
    llm_model: Optional[Callable] = None,
    **kwargs
):
    """
    Factory function to create the appropriate backend.
    
    Args:
        env_name: Environment name ("CartPole", "MountainCar", "Webshop", "FrozenLake")
        memory_strategy: Memory strategy to use:
            - "none": Original backend without memory enhancement
            - "memorybank": MemoryBank strategy (forgetting mechanism)
            - "generative": Generative strategy (LLM scoring)
            - "voyager": Voyager strategy (baseline similarity)
        storage_path: Path to experience store JSON
        depreiciate_exp_store_path: Path to deprecated store
        embedding_func: Custom embedding function (optional)
        memory_persist_dir: Custom vector DB directory (optional)
        llm_model: LLM model for scoring/summarization (optional)
        **kwargs: Additional strategy-specific parameters
        
    Returns:
        Initialized backend instance
        
    Examples:
        # Create CartPole backend with MemoryBank strategy
        backend = create_backend("CartPole", "memorybank", llm_model=my_llm)
        
        # Create MountainCar backend with Generative strategy
        backend = create_backend("MountainCar", "generative", llm_model=my_llm)
        
        # Create Webshop backend without memory (original)
        backend = create_backend("Webshop", "none")
    """
    env_name = env_name.lower()
    memory_strategy = memory_strategy.lower()
    
    # Set default paths if not provided
    if storage_path is None:
        if memory_strategy == "none":
            storage_path = f"./storage/{env_name}_exp_store.json"
        else:
            storage_path = f"./storage/{env_name}_{memory_strategy}_exp_store.json"
    
    if depreiciate_exp_store_path is None:
        if memory_strategy == "none":
            depreiciate_exp_store_path = f"./storage/{env_name}_deprecated_exp_store.json"
        else:
            depreiciate_exp_store_path = f"./storage/{env_name}_{memory_strategy}_deprecated_exp_store.json"
    
    # Mapping of (env, strategy) to backend class
    backend_map = {
        ("cartpole", "memorybank"): CartPoleMemoryBankBackend,
        ("cartpole", "generative"): CartPoleGenerativeBackend,
        ("cartpole", "voyager"): CartPoleVoyagerBackend,
        ("cartpole", "none"): CartPoleExpBackend,
        
        ("mountaincar", "memorybank"): MountainCarMemoryBankBackend,
        ("mountaincar", "generative"): MountainCarGenerativeBackend,
        ("mountaincar", "voyager"): MountainCarVoyagerBackend,
        ("mountaincar", "none"): MountainCarExpBackend,
        
        ("webshop", "memorybank"): WebshopMemoryBankBackend,
        ("webshop", "generative"): WebshopGenerativeBackend,
        ("webshop", "voyager"): WebshopVoyagerBackend,
        ("webshop", "none"): WebshopExpBackend,
        
        ("frozenlake", "memorybank"): FrozenLakeMemoryBankBackend,
        ("frozenlake", "generative"): FrozenLakeGenerativeBackend,
        ("frozenlake", "voyager"): FrozenLakeVoyagerBackend,
        ("frozenlake", "none"): FrozenLakeExpBackend,
    }
    
    key = (env_name, memory_strategy)
    if key not in backend_map:
        raise ValueError(
            f"Unknown combination: env_name={env_name}, memory_strategy={memory_strategy}\n"
            f"Supported environments: CartPole, MountainCar, Webshop, FrozenLake\n"
            f"Supported strategies: none, memorybank, generative, voyager"
        )
    
    backend_class = backend_map[key]
    
    # Create backend with appropriate parameters
    if memory_strategy == "none":
        # Original backend - simpler initialization
        backend = backend_class(
            env_name=env_name.capitalize(),
            storage_path=storage_path,
            depreiciate_exp_store_path=depreiciate_exp_store_path
        )
    else:
        # Memory-enhanced backend
        backend = backend_class(
            env_name=env_name.capitalize(),
            storage_path=storage_path,
            depreiciate_exp_store_path=depreiciate_exp_store_path,
            embedding_func=embedding_func,
            memory_persist_dir=memory_persist_dir,
            llm_model=llm_model,
            **kwargs
        )
    
    return backend


def list_available_backends():
    """
    List all available backend configurations.
    
    Returns:
        Dictionary mapping (env, strategy) to backend class
    """
    backends = {}
    
    for env in ["CartPole", "MountainCar", "Webshop", "FrozenLake"]:
        backends[env] = {}
        for strategy in ["none", "memorybank", "generative", "voyager"]:
            try:
                backend = create_backend(env, strategy)
                backends[env][strategy] = {
                    "class": backend.__class__.__name__,
                    "description": backend.__class__.__doc__.strip().split('\n')[0] if backend.__class__.__doc__ else ""
                }
                # Close the backend (clean up resources)
                if hasattr(backend, 'logIO'):
                    backend.logIO.close()
            except Exception as e:
                backends[env][strategy] = {"error": str(e)}
    
    return backends


if __name__ == "__main__":
    # Demo: list all available backends
    import json
    
    print("Available Backend Configurations:")
    print("=" * 80)
    
    backends = list_available_backends()
    print(json.dumps(backends, indent=2))
