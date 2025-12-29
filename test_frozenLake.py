import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from explorer import Explorer

# model_name = "llama3.1"
# env_name = "frozenlake_llama"
model_name = "qwen3-30b"
env_name = "frozenlake_qwen"
backend_env = "frozenlake-vanilla"
max_steps = 40
use_global_verifier = True
use_experience = True
save_experience = True
threshold =  0.25
decay_rate =  100
ts = 0

big_map = [
    "SFFHHH",
    "HHFHHH",
    "HHFHHH",
    "HHFFFG",
    "HHFHHH",
    "HHFFFG",
]

gr_group_0 = [
    {
        (3, 5): 1.0,
        (5, 5): 0.5,
    },
    {
        (3, 5): 0.5,
        (5, 5): 1.0,
    }
]

gr_group_1= [
    {
        (3, 5): 0.5,
        (5, 5): 1.0,
    },
    {
        (3, 5): 1.0,
        (5, 5): 0.5,
    }
]

# TODO: To Change
gr_group = gr_group_1


cur_name =f"log_{use_global_verifier}_{model_name}_{env_name}_{backend_env}"
log_dir=f"./log/"
backend_log_dir=log_dir
storage_path=f"./storage/exp_store.json"
depreiciate_exp_store_path=f"./storage/depreiciate_exp_store.json"


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
    desc = big_map,
    goal_rewards=gr_group[0],
)
for i in range(20):
    print(f"--- {i}/20 ---")
    e.explore()
    # Experience refinement is now handled automatically inside explore()

e.init_after_model(
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
    desc = big_map,
    goal_rewards=gr_group[1],
)
for i in range(20):
    print(f"--- {i}/20 ---")
    e.explore()
    # Experience refinement is now handled automatically inside explore()
