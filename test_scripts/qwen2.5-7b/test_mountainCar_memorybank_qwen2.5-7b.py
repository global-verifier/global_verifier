import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from explorer import Explorer

# start_timestep = 0
model_name = "qwen2.5-7b"
env_name = "mountaincar_qwen"
backend_env = "mountaincar-memorybank"

max_steps = 200
forces = [0.0016, 0.00159, 0.00158]
threshold =  0.25
decay_rate =  900
ts = 0

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
    start_timestep = ts,
    threshold = threshold,
    decay_rate = decay_rate,
    log_dir=log_dir,
    backend_log_dir=backend_log_dir,
    storage_path=storage_path,
    depreiciate_exp_store_path=depreiciate_exp_store_path,
)
for force in forces:
    if "memorybank" in backend_env:
        ts = e.exp_backend.export_status().get("mb_current_timestep")
        print(f"[MemoryBank] current timestep: {ts}")
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
        force = force,
    )

    for i in range(20):
        print(f"--- {i}/20 ---")
        e.explore()
