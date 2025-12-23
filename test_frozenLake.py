import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from explorer import Explorer
from env_adaptors.frozenLake_adaptor import FrozenLakeAdaptor

# start_timestep = 0
model_name = "llama3.1"
env_name = "frozenlake_llama"
backend_env = "frozenlake-generative"
max_steps = 20
use_global_verifier = False
use_experience = True
save_experience = True
threshold =  0.25
decay_rate =  100
ts = 0
map_0 = ["SHHH","FHHH","FFFF","HHHG"]
map_1 = ["SFHH","HFHH","FFFF","HHHG"]
map_2 = ["SFFH","HHFH","FFFF","HHHG"]

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

for i in range(20):
    print(f"--- {i}/20 ---")
    e.explore()
    # Experience refinement is now handled automatically inside explore()

if "memorybank" in backend_env:
    ts = e.exp_backend.export_status()['mb_current_timestep']
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
    desc = map_1,
)

for i in range(20):
    print(f"--- {i}/20 ---")
    e.explore()
    # Experience refinement is now handled automatically inside explore()

if "memorybank" in backend_env:
    ts = e.exp_backend.export_status()['mb_current_timestep']
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
    desc = map_2,
)

for i in range(20):
    print(f"--- {i}/20 ---")
    e.explore()
    # Experience refinement is now handled automatically inside explore()
