#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval import get_dataset, get_episodes_length  # noqa: E402


def build_cfg(config_name: str, overrides: list[str]):
    config_dir = REPO_ROOT / "config" / "eval"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name=config_name, overrides=overrides)


def sample_valid_rows(cfg, seed: int) -> tuple[list[int], int]:
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]

    g = np.random.default_rng(seed)
    row_indices = np.sort(
        g.choice(valid_indices, size=cfg.eval.num_eval, replace=False)
    ).tolist()
    return row_indices, int(valid_mask.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-groups", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override applied before sampling; can be passed multiple times.",
    )
    args = parser.parse_args()

    cfg = build_cfg(args.config_name, args.override)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num_groups):
        seed = args.seed_start + i
        row_indices, valid_count = sample_valid_rows(cfg, seed)
        payload = {
            "config_name": args.config_name,
            "overrides": args.override,
            "seed": seed,
            "num_eval": int(cfg.eval.num_eval),
            "valid_count": valid_count,
            "row_indices": row_indices,
        }
        out_path = out_dir / f"group_{i:02d}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"{out_path}: seed={seed} valid_count={valid_count}")


if __name__ == "__main__":
    main()
