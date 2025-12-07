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
    "algorithm": "sameSt_1Step",
}
