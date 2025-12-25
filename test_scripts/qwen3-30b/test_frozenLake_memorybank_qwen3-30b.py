import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from explorer import Explorer

model_name = "qwen3-30b"
env_name = "frozenlake_qwen"
backend_env = "frozenlake-memorybank"
max_steps = 20
use_global_verifier = False
use_experience = True
save_experience = True
threshold =  0.61
decay_rate =  100
ts = 0
map_0 = ["SHHH","FHHH","FFFF","HHHG"]
map_1 = ["SFHH","HFHH","FFFF","HHHG"]
map_2 = ["SFFH","HHFH","FFFF","HHHG"]

cur_name =f"log_{use_global_verifier}_{model_name}_{env_name}_{backend_env}"
log_dir=f"./{cur_name}/log/"
backend_log_dir=log_dir
storage_path=f"./{cur_name}/storage/exp_store.json"
depreiciate_exp_store_path=f"./{cur_name}/storage/depreiciate_exp_store.json"


e = Explorer(
    start_timestep = ts,
    model_name = model_name,
    env_name = env_name,
    backend_env = backend_env,
    max_steps = max_steps,
    use_global_verifier = use_global_verifier,
    use_experience = use_experience,
    save_experience = save_experience,
    threshold = threshold,
    decay_rate = decay_rate,
    desc = map_0,
)
for map in [map_0, map_1, map_2]:
    e.init_after_model(
        model_name = model_name,
        env_name = env_name,
        backend_env = backend_env,
        max_steps = max_steps,
        use_global_verifier = use_global_verifier,
        use_experience = use_experience,
        save_experience = save_experience,
        start_timestep = ts,
        threshold = threshold,
        decay_rate = decay_rate,
        log_dir=log_dir,
        backend_log_dir=backend_log_dir,
        storage_path=storage_path,
        depreiciate_exp_store_path=depreiciate_exp_store_path,
        desc = map,
    )

    for i in range(20):
        print(f"--- {i}/20 ---")
        e.explore()
        # Experience refinement is now handled automatically inside explore()
