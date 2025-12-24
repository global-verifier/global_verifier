import os
import sys

# JVM/Pyserini bootstrap：使用完整 JDK21，解决 jdk.incubator.vector 与类版本问题
_JDK_HOME = "/usr/lib/jvm/java-21-openjdk-amd64"
_JVM_PATH = os.path.join(_JDK_HOME, "lib", "server", "libjvm.so")
if os.path.exists(_JVM_PATH):
    os.environ["JAVA_HOME"] = _JDK_HOME
    os.environ["JDK_HOME"] = _JDK_HOME
    os.environ["PATH"] = f"{_JDK_HOME}/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{_JDK_HOME}/lib/server"
    os.environ["JVM_PATH"] = _JVM_PATH
    try:
        import jnius_config

        jnius_config.set_options(
            "--add-modules=jdk.incubator.vector",
            f"-Djava.home={_JDK_HOME}",
            f"-Djava.library.path={_JDK_HOME}/lib/server",
        )
        print(
            f"[pyserini jvm setup] JAVA_HOME={_JDK_HOME}, "
            f"JVM_PATH={_JVM_PATH}, python_prefix={sys.prefix}"
        )
    except Exception as e:
        print(f"[pyserini jvm setup] failed to set jnius_config: {e}")
else:
    print(f"[pyserini jvm setup] expected JVM at {_JVM_PATH} not found")

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from explorer import Explorer

# start_timestep = 0
model_name = "qwen2-7b"
env_name = "webshop_qwen"
backend_env = "webshop-vanilla"

max_steps = 20
threshold =  0.25
decay_rate =  300
start_timestep = 0

use_global_verifier = True
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
    start_timestep = start_timestep,
    threshold = threshold,
    decay_rate = decay_rate,
    log_dir=log_dir,
    backend_log_dir=backend_log_dir,
    storage_path=storage_path,
    depreiciate_exp_store_path=depreiciate_exp_store_path,
)

for i in range(20):
    print(f"--- {i}/20 ---")
    e.explore()
