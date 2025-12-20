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
    "threshold": 0.3,      # 遗忘阈值，低于此值的记忆被过滤
    "decay_rate": 8,       # 衰减速率，越大记忆保持越久 (decay_rate=8 约 10 步后移除)
}
