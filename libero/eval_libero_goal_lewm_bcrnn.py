#!/usr/bin/env python3
"""Evaluate a LeWM-BCRNN LIBERO policy with official LIBERO success checks."""

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


def make_cfg(args: argparse.Namespace, image_keys: list[str]) -> EasyDict:
    obs_key_mapping = {
        "agentview_rgb": "agentview_image",
        "eye_in_hand_rgb": "robot0_eye_in_hand_image",
        "gripper_states": "robot0_gripper_qpos",
        "joint_states": "robot0_joint_pos",
    }
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
                        "rgb": image_keys,
                        "depth": [],
                        "low_dim": ["gripper_states", "joint_states"],
                    }
                },
                "obs_key_mapping": obs_key_mapping,
            },
            "eval": {
                "n_eval": args.n_eval,
                "max_steps": args.max_steps,
                "num_procs": max(1, min(args.num_procs, args.n_eval)),
                "use_mp": args.num_procs > 1,
                "save_sim_states": False,
            },
            "lifelong": {"algo": "LeWMBCRNN"},
        }
    )


def build_task_onehot(task_id: int, batch: int, device: torch.device) -> torch.Tensor:
    task = torch.zeros((batch, 10), dtype=torch.float32, device=device)
    task[:, task_id] = 1.0
    return task


class EvalLeWMLiberoBCRNNPolicy(torch.nn.Module):
    def __init__(self, payload: dict[str, Any], device: torch.device):
        super().__init__()
        prepend(SCRIPTS_DIR)
        prepend(LEWM_CODE)
        from train_libero_goal_lewm_bcrnn import LeWMLiberoBCRNNHead  # noqa: WPS433

        self.world_model = payload["world_model"]
        head_config = dict(payload["head_config"])
        self.seq_len = int(head_config.pop("seq_len", 10))
        self.image_keys = list(head_config.pop("image_keys", ["agentview_rgb"]))
        self.trained_task_ids = list(head_config.pop("task_ids", list(range(10))))
        self.head = LeWMLiberoBCRNNHead(**head_config)
        self.head.load_state_dict(payload["head_state_dict"])
        self.task_id = 0
        self.pixel_history: dict[str, deque[torch.Tensor]] | None = None
        self.proprio_history: deque[torch.Tensor] | None = None
        self.register_buffer("action_mean", torch.as_tensor(payload["action_mean"], dtype=torch.float32).view(1, 1, -1))
        self.register_buffer("action_std", torch.as_tensor(payload["action_std"], dtype=torch.float32).view(1, 1, -1))
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
        self.pixel_history = None
        self.proprio_history = None

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

    def _encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim == 5:
            return self.world_model.encode({"pixels": pixels})["emb"]
        if pixels.ndim != 6:
            raise ValueError(f"Expected pixels with 5 or 6 dims, got {tuple(pixels.shape)}")
        batch, seq_len, n_views, channels, height, width = pixels.shape
        flat = pixels.permute(0, 2, 1, 3, 4, 5).reshape(batch * n_views, seq_len, channels, height, width)
        emb = self.world_model.encode({"pixels": flat})["emb"]
        return emb.reshape(batch, n_views, seq_len, emb.shape[-1]).permute(0, 2, 1, 3).reshape(batch, seq_len, -1)

    def get_action(self, data: dict[str, Any]) -> np.ndarray:
        current_pixels = {key: self._prep_pixels(data["obs"][key]) for key in self.image_keys}
        batch = next(iter(current_pixels.values())).shape[0]
        proprio = torch.cat(
            [
                data["obs"]["joint_states"].to(self.device).float(),
                data["obs"]["gripper_states"].to(self.device).float(),
            ],
            dim=-1,
        )
        proprio = (proprio - self.proprio_mean) / self.proprio_std
        if self.pixel_history is None:
            self.pixel_history = {key: deque([value] * self.seq_len, maxlen=self.seq_len) for key, value in current_pixels.items()}
            self.proprio_history = deque([proprio] * self.seq_len, maxlen=self.seq_len)
        else:
            for key, value in current_pixels.items():
                self.pixel_history[key].append(value)
            self.proprio_history.append(proprio)
        stacked_pixels = torch.stack(
            [torch.stack(list(self.pixel_history[key]), dim=1) for key in self.image_keys],
            dim=2,
        )
        stacked_proprio = torch.stack(list(self.proprio_history), dim=1)
        task = build_task_onehot(self.task_id, batch, self.device).unsqueeze(1).expand(-1, self.seq_len, -1)
        with torch.no_grad():
            emb = self._encode_pixels(stacked_pixels)
            normalized = self.head(emb, stacked_proprio, task).float()
            action = normalized[:, -1:] * self.action_std + self.action_mean
        return np.clip(action[:, 0].detach().cpu().numpy(), -1.0, 1.0)


class AlgoWrapper(torch.nn.Module):
    def __init__(self, policy: EvalLeWMLiberoBCRNNPolicy):
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
    policy = EvalLeWMLiberoBCRNNPolicy(payload, device)
    train_args = dict(payload.get("args", {}))
    train_encoder = bool(train_args.get("train_encoder", False))
    algo = AlgoWrapper(policy)
    cfg = make_cfg(args, policy.image_keys)
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
        print(f"[libero-lewm-bcrnn] task={task_id} success={success_rate:.3f}", flush=True)

    result = {
        "classification": (
            "rcaux_libero_goal_lewm_trainable_encoder_bcrnn_official_eval"
            if train_encoder
            else "rcaux_libero_goal_lewm_frozen_encoder_bcrnn_official_eval"
        ),
        "checkpoint": str(args.checkpoint),
        "train_encoder": train_encoder,
        "trained_task_ids": policy.trained_task_ids,
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
        args.output = ROOT / "results" / f"libero_goal_rcaux_lewm_bcrnn_eval_tasks_{label}_n{args.n_eval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
