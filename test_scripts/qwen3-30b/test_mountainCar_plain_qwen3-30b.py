import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from explorer import Explorer

# start_timestep = 0
model_name = "qwen3-30b"
env_name = "mountaincar_qwen"
backend_env = "mountaincar-vanilla"

max_steps = 200
forces = [0.0016, 0.00159, 0.00158]

use_global_verifier = False
use_experience = True
save_experience = True

cur_name =f"log_{use_global_verifier}_{model_name}_{env_name}_{backend_env}"
log_dir=f"./{cur_name}/log/"
backend_log_dir=log_dir
storage_path=f"./{cur_name}/storage/exp_store.json"
depreiciate_exp_store_path=f"./{cur_name}/storage/depreiciate_exp_store.json"


e = Explorer(
    model_name = model_name,
    env_name = env_name,
    backend_env = backend_env,
    max_steps = max_steps,
    use_global_verifier = use_global_verifier,
    use_experience = use_experience,
    save_experience = save_experience,
    log_dir=log_dir,
    backend_log_dir=backend_log_dir,
    storage_path=storage_path,
    depreiciate_exp_store_path=depreiciate_exp_store_path,
)
for force in forces:
    e.init_after_model(
        model_name = model_name,
        env_name = env_name,
        backend_env = backend_env,
        max_steps = max_steps,
        use_global_verifier = use_global_verifier,
        use_experience = use_experience,
        save_experience = save_experience,
        log_dir=log_dir,
        backend_log_dir=backend_log_dir,
        storage_path=storage_path,
        depreiciate_exp_store_path=depreiciate_exp_store_path,
        force = force,
    )

    for i in range(20):
        print(f"--- {i}/20 ---")
        e.explore()
        # Experience refinement is now handled automatically inside explore()
