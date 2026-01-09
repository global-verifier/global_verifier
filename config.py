explorer_settings = {
    "max_steps": 20,  # Mountain Car needs more steps!
    "max_action_retries": 3,
    "log_dir": "./log/",
    # Experience settings
    # Unified memory toggle (preferred)
    "use_memory": True,
    "use_experience": True,   # Whether to retrieve and use experiences in prompts
    "save_experience": True,  # Whether to save new experiences to storage
    "use_global_verifier": False,  # Whether to use global verifier (refine_experience) after each explore
    "conflict_soultion": "st", # "st" or "conflict"
    # Plug-in settings
    "model_name": "llama3.1",
    "env_name": "frozenlake",  # Change env
    "backend_env": "frozenlake-memorybank",  # Change backend
    "storage_path": "./storage/exp_store.json",
    "depreiciate_exp_store_path": "./storage/depreiciate_exp_store.json",
    "alpha": 10,
}
model_path = {
    # llama3 models
    # "llama3-8b": "/data/xingkun/local_model/Meta-Llama-3-8B-Instruct",
    "llama3.1-8b": "/data/xingkun/local_model/Meta-Llama-3.1-8B-Instruct",
    # "llama3.2-3b": "/data/xingkun/local_model/Llama-3.2-3B-Instruct",
    "llama3.3-70b": "/data/xingkun/local_model/Llama-3.3-70B-Instruct",
    # qwen models
    # "qwen2-7b": "/data/xingkun/local_model/Qwen2-7B-Instruct",
    "qwen2.5-7b": "/data/xingkun/local_model/Qwen2.5-7B-Instruct",
    "qwen3-30b": "/data/xingkun/local_model/Qwen3-30B-A3B-Instruct-2507",
    # mistral models
    # "mistral3-14b": "/data/xingkun/local_model/Ministral-3-14B-Instruct-2512",
    # "mistral3.2-24b": "/data/xingkun/local_model/Mistral-Small-3.2-24B-Instruct-2506",
    # internlm models
    # "internlm3-8b": "/data/xingkun/local_model/internlm3-8b-instruct",
    # deepseek models
    # "deepseek-v2": "/data/xingkun/local_model/DeepSeek-Coder-V2-Lite-Instruct",
}
api_model_name = {
    # openai models
    "gpt-4o": "gpt-4o",
    # llama
    "llama-3.3-70b-instruct": "llama-3.3-70b-instruct",
    # gemini
    "gemini-2.0-flash": "gemini-2.0-flash",
    # deepseek
    "deepseek-chat": "deepseek-chat",
    "deepseek-r1": "deepseek-r1",  # ok, but slow
    # grok
    "grok-3": "grok-3",
    # qwen
    "qwen3-235b-a22b": "qwen3-235b-a22b",
    "qwen3-235b-a22b-instruct-2507": "qwen3-235b-a22b-instruct-2507",
    "qwen2.5-7b-instruct": "qwen2.5-7b-instruct",
    # claude models
    "claude-3-5-haiku-20241022": "claude-3-5-haiku-20241022",
    # "gpt-5-nano": "gpt-5-nano",  # no
    # "deepseek-v3": "deepseek-v3",
}
api = {
    "base_url": "https://hk.yi-zhan.top/v1",
    # "base_url": "https://openrouter.ai/api/v1",
}
