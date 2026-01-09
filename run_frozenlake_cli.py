#!/usr/bin/env python3
"""
CLI runner for FrozenLake experiments.

This script is meant to be convenient to run from any Linux terminal / cwd.
It mirrors `run_frozenLake.py` but accepts key configs from CLI.
"""

import argparse
import os
import sys
from typing import Any


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
        description="Run FrozenLake exploration with configurable flags from CLI."
    )

    # Required by user
    p.add_argument("--use-memory", type=str2bool, required=True)
    p.add_argument(
        "--memory-env",
        type=str,
        required=True,
        choices=["vanilla", "generative", "memorybank", "voyager"],
        help="Memory backend mode. Mirrors Explorer.process_memory_env().",
    )
    p.add_argument("--model-name", type=str, required=True)

    # Optional QoL flags (defaults mirror `run_frozenLake.py`)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--decay-rate", type=float, default=60.0)
    p.add_argument("--episodes-per-map", type=int, default=20)
    p.add_argument("--output-root", type=str, default=".")
    p.add_argument("--cuda-visible-devices", type=str, default=None)
    p.add_argument("--use-global-verifier", type=str2bool, default=None)
    p.add_argument(
        "--use-api",
        type=str2bool,
        default=True,
        help="Whether to use API model backend when loading the explorer model.",
    )
    return p


def main() -> int:
    args = build_argparser().parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    # Make imports work no matter where user runs this from.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from explorer import Explorer  # noqa: E402

    map_0 = [
        "SFHHHH",
        "HFFHHH",
        "HHFFHH",
        "HHHFFH",
        "HHHHFF",
        "HHHHHG",
    ]
    map_1 = [
        "SFHHHH",
        "HFHHHH",
        "HFFFFH",
        "HHHHFH",
        "HHHHFH",
        "HHHHFG",
    ]
    # map_1 = [
    #     "SFFFHH",
    #     "HHHFFH",
    #     "HHHHFF",
    #     "HHHHHF",
    #     "HHHHHF",
    #     "HHHHHG",
    # ]
    map_2 =[
        "SFHHHH",
        "HFHHHH",
        "HFHHHH",
        "HFFHHH",
        "HHFFHH",
        "HHHFFG",
    ]

    maps_to_run = [map_0, map_1, map_2]
    env_name = "frozenlake"

    cur_name = f"log_{env_name}_{args.model_name}_{args.memory_env}_{args.use_memory}_{args.use_global_verifier}"
    run_root = os.path.join(args.output_root, cur_name)
    log_dir = os.path.join(run_root, "log")
    backend_log_dir = log_dir
    storage_path = os.path.join(run_root, "storage", "exp_store.json")
    depreiciate_exp_store_path = os.path.join(run_root, "storage", "depreiciate_exp_store.json")

    # Initialize once (model load happens here); per-map we call init_after_model to avoid reload.
    ts = 0
    e = Explorer(
        model_name=args.model_name,
        env_name=env_name,
        memory_env=args.memory_env,
        max_steps=args.max_steps,
        use_memory=args.use_memory,
        start_timestep=ts,
        threshold=args.threshold,
        decay_rate=args.decay_rate,
        log_dir=log_dir,
        backend_log_dir=backend_log_dir,
        storage_path=storage_path,
        depreiciate_exp_store_path=depreiciate_exp_store_path,
        desc=maps_to_run[0],
        use_api=args.use_api,
        use_global_verifier=args.use_global_verifier,
    )

    for map_idx, cur_map in enumerate(maps_to_run):
        # MemoryBank keeps an integer timestep for forgetting; when we re-init the backend
        # (switching maps), carry forward the latest timestep so forgetting continues.
        status = e.exp_backend.export_status()
        if status is not None:
            ts = status.get("mb_current_timestep", ts)
        e.init_after_model(
            model_name=args.model_name,
            env_name=env_name,
            memory_env=args.memory_env,
            max_steps=args.max_steps,
            use_memory=args.use_memory,
            start_timestep=ts,
            threshold=args.threshold,
            decay_rate=args.decay_rate,
            log_dir=log_dir,
            backend_log_dir=backend_log_dir,
            storage_path=storage_path,
            depreiciate_exp_store_path=depreiciate_exp_store_path,
            desc=cur_map,
            use_global_verifier=args.use_global_verifier,
        )

        for i in range(args.episodes_per_map):
            print(f"--- map {map_idx} | episode {i}/{args.episodes_per_map} ---")
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

# python3 /home/xingkun/global_verifier/run_frozenlake_cli.py \
#   --use-memory true \
#   --model-name deepseek-chat \
#   --memory-env vanilla \
#   --use-global-verifier true

# python3 /home/xingkun/global_verifier/run_frozenlake_cli.py \
#   --cuda-visible-devices 4 \
#   --use-memory true \
#   --model-name deepseek-chat \
#   --memory-env memorybank \
#   --use-global-verifier true
