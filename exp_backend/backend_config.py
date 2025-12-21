base_backend_config = {
    "log_dir": "./log",
}
webshop_vanilla_config = {
    "algorithm": "sameSt_bfs_maxScore",  # “sameSt_1Step” or "sameSt_bfs_maxScore"
    "max_bfs_depth": 10,  # Only used when algorithm is "sameSt_bfs_maxScore"
}
frozenlake_vanilla_config = {
    "algorithm": "sameSt_1Step",
}
cartpole_vanilla_config = {
    "algorithm": "sameSt_1Step",
}
mountaincar_vanilla_config = {
    "algorithm": "sameSt_bfs_reachable",  # "sameSt_1Step" or "sameSt_bfs_reachable"
    "max_bfs_depth": 200,  # Maximum BFS search depth (MountainCar may need many steps)
    "goal_position": 0.5,  # Goal position threshold
}
memorybank_config = {
    "threshold": 0.2,      # 遗忘阈值，低于此值的记忆被过滤
    "decay_rate": 200,       # 衰减速率，越大记忆保持越久
}
voyager_config = {
    "model_path": None,    # LLM 模型路径，None 则使用外部传入的 llm_func
    "max_new_tokens": 128, # 生成总结的最大 token 数
}
