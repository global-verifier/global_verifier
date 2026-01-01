#!/usr/bin/env python3
"""
CLI runner for Webshop experiments.

This mirrors `test_webshop.py` but exposes key settings via CLI so it can be
invoked from any cwd without manual edits.
"""

import argparse
import os
import sys
from typing import Any

# JVM/Pyserini bootstrap: mirror `test_webshop.py` to load jdk.incubator.vector
_JDK_HOME = "/usr/lib/jvm/java-21-openjdk-amd64"
_JVM_PATH = os.path.join(_JDK_HOME, "lib", "server", "libjvm.so")
if os.path.exists(_JVM_PATH):
    os.environ["JAVA_HOME"] = _JDK_HOME
    os.environ["JDK_HOME"] = _JDK_HOME
    os.environ["PATH"] = f"{_JDK_HOME}/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{_JDK_HOME}/lib/server"
    os.environ["JVM_PATH"] = _JVM_PATH
    try:
        import jnius_config  # type: ignore

        jnius_config.set_options(
            "--add-modules=jdk.incubator.vector",
            f"-Djava.home={_JDK_HOME}",
            f"-Djava.library.path={_JDK_HOME}/lib/server",
        )
        print(
            f"[pyserini jvm setup] JAVA_HOME={_JDK_HOME}, "
            f"JVM_PATH={_JVM_PATH}, python_prefix={sys.prefix}"
        )
    except Exception as e:  # pragma: no cover - best-effort boot
        print(f"[pyserini jvm setup] failed to set jnius_config: {e}")
else:  # pragma: no cover - best-effort boot
    print(f"[pyserini jvm setup] expected JVM at {_JVM_PATH} not found")


def str2bool(v: Any) -> bool:
    """Parse common boolean strings from CLI."""
    if isinstance(v, bool):
        return v
    if v is None:
        raise argparse.ArgumentTypeError("Boolean value expected, got None")
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v!r}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Webshop exploration with configurable flags from CLI."
    )

    # Required toggles
    p.add_argument("--use-memory", type=str2bool, required=True)
    p.add_argument(
        "--memory-env",
        type=str,
        required=True,
        choices=["vanilla", "generative", "memorybank", "voyager", "glove"],
        help="Memory backend mode. Mirrors Explorer.process_memory_env().",
    )
    p.add_argument("--model-name", type=str, required=True)

    # Optional QoL flags (defaults mirror `test_webshop.py` where applicable)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--decay-rate", type=float, default=300.0)
    p.add_argument("--start-timestep", type=int, default=0)
    p.add_argument("--episodes", type=int, default=20, help="Episodes to run.")
    p.add_argument("--output-root", type=str, default=".")
    p.add_argument("--cuda-visible-devices", type=str, default=None)
    p.add_argument(
        "--enable-confirm-purchase",
        type=str2bool,
        # TODO: whether to change to false
        default=True,
        help="Whether to enable confirm purchase flow (webshop-specific).",
    )
    p.add_argument(
        "--use-api",
        type=str2bool,
        default=True,
        help="Whether to use API model backend when loading the explorer model.",
    )
    return p


def main() -> int:
    # warm color
    session = 9
    
    args = build_argparser().parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    # Make imports work no matter where user runs this from.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from explorer import Explorer  # noqa: E402

    env_name = "webshop"

    cur_name = f"log_{args.use_memory}_{env_name}_{args.memory_env}_{args.model_name}"
    run_root = os.path.join(args.output_root, cur_name)
    log_dir = os.path.join(run_root, "log")
    backend_log_dir = log_dir
    storage_path = os.path.join(run_root, "storage", "exp_store.json")
    depreiciate_exp_store_path = os.path.join(
        run_root, "storage", "depreiciate_exp_store.json"
    )

    # Initialize once (model load happens here).
    e = Explorer(
        model_name=args.model_name,
        env_name=env_name,
        memory_env=args.memory_env,
        max_steps=args.max_steps,
        use_memory=args.use_memory,
        start_timestep=args.start_timestep,
        threshold=args.threshold,
        decay_rate=args.decay_rate,
        log_dir=log_dir,
        backend_log_dir=backend_log_dir,
        storage_path=storage_path,
        depreiciate_exp_store_path=depreiciate_exp_store_path,
        enable_confirm_purchase=args.enable_confirm_purchase,
        session=session,
        use_api=args.use_api,
    )

    for i in range(args.episodes):
        print(f"--- episode {i}/{args.episodes} ---")
        e.explore()

    # Create a finish marker file to indicate this run completed successfully.
    marker_dir = os.path.join(args.output_root, "finish_mark")
    os.makedirs(marker_dir, exist_ok=True)
    marker_path = os.path.join(marker_dir, cur_name)
    with open(marker_path, "w", encoding="utf-8"):
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Example:
# python run_webshop_cli.py --use-memory true --model-name openai-gpt-3.5-turbo-instruct --memory-env glove

