explorer_settings = {
    "max_steps": 20,
    "max_action_retries": 3,
    "log_dir": "./log/",
    # Experience settings
    "use_experience": True,   # Whether to retrieve and use experiences in prompts
    "save_experience": False,  # Whether to save new experiences to storage
    # Plug-in settings
    "model_name": "llama3.1",
    "env_name": "webshop_llama",
    "backend_env": "webshop-vanilla",
    "storage_path": "./storage/exp_store.json",
}
model_path = {
    "llama3": "/data/xingkun/local_model/Meta-Llama-3-8B-Instruct",
    "llama3.1": "/data/xingkun/local_model/Meta-Llama-3.1-8B-Instruct",
    "qwen2.5": "/data/xingkun/local_model/Qwen2.5-7B-Instruct",
    "qwen3": "/data/xingkun/local_model/Qwen3-8B",
}
