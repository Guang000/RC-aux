#!/usr/bin/env python3
"""Evaluate a LeWM-BC LIBERO policy with official LIBERO success checks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict


ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = ROOT / "assets" / "benchmarks" / "LIBERO"
LEWM_CODE = ROOT
SCRIPTS_DIR = ROOT / "libero"


def prepend(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def parse_task_ids(value: str) -> list[int]:
    if value.lower() == "all":
        return list(range(10))
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--n-eval", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--num-procs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def make_cfg(args: argparse.Namespace) -> EasyDict:
    return EasyDict(
        {
            "device": args.device,
            "bddl_folder": str(LIBERO_ROOT / "libero" / "libero" / "bddl_files"),
            "init_states_folder": str(LIBERO_ROOT / "libero" / "libero" / "init_files"),
            "data": {
                "img_h": 128,
                "img_w": 128,
                "obs": {
                    "modality": {
                        "rgb": ["agentview_rgb"],
                        "depth": [],
                        "low_dim": ["gripper_states", "joint_states"],
                    }
                },
                "obs_key_mapping": {
                    "agentview_rgb": "agentview_image",
                    "gripper_states": "robot0_gripper_qpos",
                    "joint_states": "robot0_joint_pos",
                },
            },
            "eval": {
                "n_eval": args.n_eval,
                "max_steps": args.max_steps,
                "num_procs": max(1, min(args.num_procs, args.n_eval)),
                "use_mp": args.num_procs > 1,
                "save_sim_states": False,
            },
            "lifelong": {"algo": "LeWMBC"},
        }
    )


def build_task_onehot(task_id: int, batch: int, device: torch.device) -> torch.Tensor:
    task = torch.zeros((batch, 10), dtype=torch.float32, device=device)
    task[:, task_id] = 1.0
    return task


class EvalLeWMLiberoBCPolicy(torch.nn.Module):
    def __init__(self, payload: dict[str, Any], device: torch.device):
        super().__init__()
        prepend(SCRIPTS_DIR)
        prepend(LEWM_CODE)
        from train_libero_goal_lewm_bc import LeWMLiberoBCHead  # noqa: WPS433

        self.world_model = payload["world_model"]
        self.head = LeWMLiberoBCHead(**payload["head_config"])
        self.head.load_state_dict(payload["head_state_dict"])
        self.n_obs_steps = int(payload["head_config"]["n_obs_steps"])
        self.task_id = 0
        self.history: deque[torch.Tensor] | None = None
        self.register_buffer("action_mean", torch.as_tensor(payload["action_mean"], dtype=torch.float32).view(1, -1))
        self.register_buffer("action_std", torch.as_tensor(payload["action_std"], dtype=torch.float32).view(1, -1))
        self.register_buffer("proprio_mean", torch.as_tensor(payload["proprio_mean"], dtype=torch.float32).view(1, -1))
        self.register_buffer("proprio_std", torch.as_tensor(payload["proprio_std"], dtype=torch.float32).view(1, -1))
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))
        self.to(device)
        self.eval()
        self.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def reset(self) -> None:
        self.history = None

    def set_task_id(self, task_id: int) -> None:
        self.task_id = task_id
        self.reset()

    def _prep_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels.to(self.device).float()
        if pixels.max() > 2.0:
            pixels = pixels / 255.0
        if pixels.shape[-2:] != (224, 224):
            pixels = F.interpolate(pixels, size=(224, 224), mode="bilinear", align_corners=False)
        return (pixels - self.image_mean) / self.image_std

    def get_action(self, data: dict[str, Any]) -> np.ndarray:
        pixels = self._prep_pixels(data["obs"]["agentview_rgb"])
        batch = pixels.shape[0]
        if self.history is None:
            self.history = deque([pixels] * self.n_obs_steps, maxlen=self.n_obs_steps)
        else:
            self.history.append(pixels)
            while len(self.history) < self.n_obs_steps:
                self.history.appendleft(pixels)
        stacked = torch.stack(list(self.history), dim=1)
        proprio = torch.cat(
            [
                data["obs"]["joint_states"].to(self.device).float(),
                data["obs"]["gripper_states"].to(self.device).float(),
            ],
            dim=-1,
        )
        proprio = (proprio - self.proprio_mean) / self.proprio_std
        task = build_task_onehot(self.task_id, batch, self.device)
        with torch.no_grad():
            emb = self.world_model.encode({"pixels": stacked})["emb"]
            normalized = self.head(emb, proprio, task).float()
            action = normalized * self.action_std + self.action_mean
        return np.clip(action.detach().cpu().numpy(), -1.0, 1.0)


class AlgoWrapper(torch.nn.Module):
    def __init__(self, policy: EvalLeWMLiberoBCPolicy):
        super().__init__()
        self.policy = policy

    def reset(self) -> None:
        self.policy.reset()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_ROOT / ".libero"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    prepend(LIBERO_ROOT)
    prepend(LEWM_CODE)
    prepend(SCRIPTS_DIR)

    from libero.libero import benchmark
    from libero.lifelong.metric import evaluate_one_task_success
    import robomimic.utils.obs_utils as ObsUtils

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = str(device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy = EvalLeWMLiberoBCPolicy(payload, device)
    algo = AlgoWrapper(policy)
    cfg = make_cfg(args)
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": cfg.data.obs.modality})
    bench = benchmark.get_benchmark_dict()["libero_goal"]()
    task_ids = parse_task_ids(args.tasks)

    started = time.time()
    rows = []
    for task_id in task_ids:
        policy.set_task_id(task_id)
        task = bench.get_task(task_id)
        task_emb = torch.zeros((1, 1), dtype=torch.float32, device=device)
        task_start = time.time()
        success_rate = evaluate_one_task_success(cfg, algo, task, task_emb, task_id)
        rows.append(
            {
                "task_id": task_id,
                "task_name": task.name,
                "language": task.language,
                "success_rate": float(success_rate),
                "n_eval": args.n_eval,
                "elapsed_sec": time.time() - task_start,
            }
        )
        print(f"[libero-lewm-bc] task={task_id} success={success_rate:.3f}", flush=True)

    result = {
        "classification": "rcaux_libero_goal_lewm_frozen_encoder_bc_official_eval",
        "checkpoint": str(args.checkpoint),
        "task_ids": task_ids,
        "n_eval": args.n_eval,
        "max_steps": args.max_steps,
        "mean_success_rate": float(np.mean([row["success_rate"] for row in rows])),
        "tasks": rows,
        "elapsed_sec": time.time() - started,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if args.output is None:
        label = "all" if len(task_ids) == 10 else "_".join(map(str, task_ids))
        args.output = ROOT / "results" / f"libero_goal_rcaux_lewm_bc_eval_tasks_{label}_n{args.n_eval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
