webshop_config = {
    "id": "WebAgentTextEnv-v0",
    "observation_mode": "text",
    "num_products": 1000,
    "human_goals": 1,  # 使用人类标注的任务 (1=True, 0=False)

    # session
    # Deterministic goal index (0-based). 8 -> faux fur coat instruction.
    "session": 9,
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

cartpole_config = {
    "id": "CartPole-v1",
    "random_seed": 0,
    # Optional: customize physics parameters
    # "force_mag": 10.0,      # Default: 10.0 N (push force)
    # "gravity": 9.8,         # Default: 9.8 m/s²
    # "masscart": 1.0,        # Default: 1.0 kg
    # "masspole": 0.1,        # Default: 0.1 kg
    # "length": 0.5,          # Default: 0.5 m (half-pole length)
    # "tau": 0.02,            # Default: 0.02 s (time step)
}
