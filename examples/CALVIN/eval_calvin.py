"""Run CALVIN 5-task chain (long-horizon) evaluation with our model as policy.

Usage:
    python examples/CALVIN/eval_calvin.py \
        --dataset_path /mnt/4TB_2/jamvla/datasets/calvin/task_D_D \
        --host 127.0.0.1 --port 20080 \
        --num_sequences 1000 \
        --log_dir results/eval_calvin/D
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# CALVIN model package sits in a different tree; add both to sys.path
CALVIN_ROOT = Path("/home/choi/calvin")
sys.path.insert(0, str(CALVIN_ROOT / "calvin_models"))
sys.path.insert(0, str(CALVIN_ROOT / "calvin_env"))

from calvin_agent.evaluation.evaluate_policy import evaluate_policy   # noqa
from calvin_env.envs.play_table_env import get_env                    # noqa

# Our policy wrapper
from examples.CALVIN.calvin_client import OurCalvinModel


def make_env(dataset_path: str):
    val_folder = Path(dataset_path) / "validation"
    return get_env(val_folder, show_gui=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True,
                   help="CALVIN dataset root (must contain 'validation/' subdir)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=20080)
    p.add_argument("--num_sequences", type=int, default=1000)
    p.add_argument("--log_dir", default="results/eval_calvin/run")
    p.add_argument("--epoch", default="final")
    args = p.parse_args()

    # Patch NUM_SEQUENCES if caller wants a shorter smoke run
    import calvin_agent.evaluation.evaluate_policy as ep
    ep.NUM_SEQUENCES = args.num_sequences

    os.makedirs(args.log_dir, exist_ok=True)
    print(f"[eval_calvin] dataset={args.dataset_path}")
    print(f"[eval_calvin] policy server: ws://{args.host}:{args.port}")
    print(f"[eval_calvin] num_sequences={args.num_sequences}")

    env = make_env(args.dataset_path)
    model = OurCalvinModel(host=args.host, port=args.port)

    evaluate_policy(model=model, env=env, epoch=args.epoch,
                    eval_log_dir=args.log_dir, debug=False)


if __name__ == "__main__":
    main()
