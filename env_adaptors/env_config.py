webshop_config = {
    "id": "WebAgentTextEnv-v0",
    "observation_mode": "text",
    "num_products": 1000,
    "human_goals": 1,  # 使用人类标注的任务 (1=True, 0=False)

    # session
    # Deterministic goal index (0-based). 8 -> faux fur coat instruction.
    "session": 8,
    "random_seed": 0,
}

frozenlake_config = {
    "id": "FrozenLake-v1",
    # "desc": None,
    "desc": ["SFFF","FHFH","FFFH","HFFG"],
    "random_seed": 0,
    "map_name": "4x4",
    "is_slippery": False,
    "success_rate": 1,
    "reward_schedule": (1, 0, 0),
    "max_episode_steps": None,
}
