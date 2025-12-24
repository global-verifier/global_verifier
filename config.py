explorer_settings = {
    "max_steps": 20,  # Mountain Car needs more steps!
    "max_action_retries": 3,
    "log_dir": "./log/",
    # Experience settings
    "use_experience": True,   # Whether to retrieve and use experiences in prompts
    "save_experience": True,  # Whether to save new experiences to storage
    "use_global_verifier": False,  # Whether to use global verifier (refine_experience) after each explore
    "conflict_soultion": "st", # "st" or "conflict"
    # Plug-in settings
    "model_name": "llama3.1",
    "env_name": "frozenlake_llama",  # Change env
    "backend_env": "frozenlake-memorybank",  # Change backend
    "storage_path": "./storage/exp_store.json",
    "depreiciate_exp_store_path": "./storage/depreiciate_exp_store.json",
}
model_path = {
    "llama3-8b": "/data/xingkun/local_model/Meta-Llama-3-8B-Instruct",
    "llama3.1": "/data/xingkun/local_model/Meta-Llama-3.1-8B-Instruct",
    # "llama3.2": "/data/xingkun/local_model/Llama-3.2-3B-Instruct",
    "qwen2.5-7b": "/data/xingkun/local_model/Qwen2.5-7B-Instruct",
    # "qwen3-8B": "/data/xingkun/local_model/Qwen3-8B",
    "qwen3-30b": "/data/xingkun/local_model/Qwen3-30B-A3B-Instruct-2507",
}
