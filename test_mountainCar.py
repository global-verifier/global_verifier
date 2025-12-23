import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from explorer import Explorer

# start_timestep = 0
model_name = "llama3.1"
env_name = "mountaincar_llama"
backend_env = "mountaincar-vanilla"
max_steps = 20
use_global_verifier = True
use_experience = True
save_experience = True

forces = [0.0016, 0.00159, 0.00158]

e = Explorer(
    model_name = model_name,
    env_name = env_name,
    backend_env = backend_env,
    max_steps = max_steps,
    use_global_verifier = use_global_verifier,
    use_experience = use_experience,
    save_experience = save_experience,
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
        force = force,
    )

    for i in range(20):
        print(f"--- {i}/20 ---")
        e.explore()
        # Experience refinement is now handled automatically inside explore()
