#!/usr/bin/env python3
"""Evaluate LeWM-style planners on DINO-WM family env-native benchmarks."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import types
from collections import deque
from pathlib import Path
from typing import Any

import h5py
import numpy as np


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


BENCHMARK_SPECS: dict[str, dict[str, Any]] = {
    "wall": {
        "frameskip": 5,
        "goal_source": "random_state",
        "goal_h": 5,
        "max_episode_steps": 300,
        "action_block": 5,
        "env_ctor": ("env.wall.wall_env_wrapper", "WallEnvWrapper"),
        "default_dataset": "data/dino_wall.h5",
    },
    "pusht_noise": {
        "frameskip": 5,
        "goal_source": "dset",
        "goal_h": 5,
        "max_episode_steps": 300,
        "action_block": 5,
        "env_ctor": ("env.pusht.pusht_wrapper", "PushTWrapper"),
        "default_dataset": "data/dino_pusht_noise_train.h5",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--benchmark", choices=tuple(BENCHMARK_SPECS), required=True)
    parser.add_argument("--policy-kind", choices=("hf", "object"), required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-rollout-dir", type=Path, default=None)
    parser.add_argument("--eval-plan", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--val-dataset-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--receding-horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=None)
    parser.add_argument(
        "--goal-cost-reduce",
        choices=("terminal", "min", "mean", "softmin"),
        default=None,
    )
    parser.add_argument("--goal-cost-softmin-temperature", type=float, default=None)
    parser.add_argument(
        "--use-reachability-cost",
        choices=("default", "on", "off"),
        default="off",
    )
    parser.add_argument("--reachability-cost-weight", type=float, default=None)
    parser.add_argument("--latent-cost-weight", type=float, default=None)
    parser.add_argument(
        "--use-temporal-distance-cost",
        choices=("default", "on", "off"),
        default="off",
    )
    parser.add_argument("--temporal-distance-cost-weight", type=float, default=None)
    parser.add_argument(
        "--temporal-distance-cost-reduce",
        choices=("terminal", "min", "mean"),
        default=None,
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def prepend_sys_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def install_d4rl_stub() -> None:
    if "d4rl" in sys.modules:
        return
    offline_env = types.ModuleType("offline_env")

    class OfflineEnv:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    offline_env.OfflineEnv = OfflineEnv
    d4rl = types.ModuleType("d4rl")
    d4rl.offline_env = offline_env
    sys.modules["d4rl"] = d4rl
    sys.modules["d4rl.offline_env"] = offline_env


def find_model_with_attr(module: Any, attr: str) -> Any | None:
    if hasattr(module, attr):
        return module.eval() if hasattr(module, "eval") else module
    for child in getattr(module, "children", lambda: [])():
        result = find_model_with_attr(child, attr)
        if result is not None:
            return result
    return None


def _resolve_tristate(flag: str) -> bool | None:
    if flag == "default":
        return None
    return flag == "on"


def apply_model_overrides(model: Any, args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.goal_cost_reduce is not None:
        model.goal_cost_reduce = args.goal_cost_reduce
        overrides["goal_cost_reduce"] = args.goal_cost_reduce
    if args.goal_cost_softmin_temperature is not None:
        model.goal_cost_softmin_temperature = args.goal_cost_softmin_temperature
        overrides["goal_cost_softmin_temperature"] = args.goal_cost_softmin_temperature

    use_reachability = _resolve_tristate(args.use_reachability_cost)
    if use_reachability is not None:
        model.use_reachability_cost = use_reachability
        overrides["use_reachability_cost"] = use_reachability
    if args.reachability_cost_weight is not None:
        model.reachability_cost_weight = args.reachability_cost_weight
        overrides["reachability_cost_weight"] = args.reachability_cost_weight
    if args.latent_cost_weight is not None:
        model.latent_cost_weight = args.latent_cost_weight
        overrides["latent_cost_weight"] = args.latent_cost_weight

    use_td = _resolve_tristate(args.use_temporal_distance_cost)
    if use_td is not None:
        model.use_temporal_distance_cost = use_td
        overrides["use_temporal_distance_cost"] = use_td
    if args.temporal_distance_cost_weight is not None:
        model.temporal_distance_cost_weight = args.temporal_distance_cost_weight
        overrides["temporal_distance_cost_weight"] = args.temporal_distance_cost_weight
    if args.temporal_distance_cost_reduce is not None:
        model.temporal_distance_cost_reduce = args.temporal_distance_cost_reduce
        overrides["temporal_distance_cost_reduce"] = args.temporal_distance_cost_reduce
    return overrides


def reset_policy_state(policy: Any) -> None:
    if getattr(policy, "_action_buffer", None) is not None:
        policy._action_buffer = deque(maxlen=policy.flatten_receding_horizon)
    if hasattr(policy, "_next_init"):
        policy._next_init = None


class NumpyStandardScaler:
    def fit(self, x: np.ndarray) -> "NumpyStandardScaler":
        x = np.asarray(x, dtype=np.float32)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_ = np.where(self.scale_ < 1e-8, 1.0, self.scale_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return x * self.scale_ + self.mean_


class _PolicyEnvAdapter:
    def __init__(self, action_space: Any):
        self.action_space = action_space
        self.num_envs = 1


def make_batched_action_space(action_space: Any) -> Any:
    import gymnasium as gym

    low = np.expand_dims(np.asarray(action_space.low), axis=0)
    high = np.expand_dims(np.asarray(action_space.high), axis=0)
    return gym.spaces.Box(low=low, high=high, dtype=action_space.dtype)


def get_or_create_action_space(env: Any) -> Any:
    import gymnasium as gym

    if hasattr(env, "action_space"):
        return env.action_space
    if hasattr(env, "action_dim"):
        dim = int(env.action_dim)
        low = -np.ones((dim,), dtype=np.float32)
        high = np.ones((dim,), dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)
    raise AttributeError(f"Env {type(env)} has neither action_space nor action_dim")


def normalize_visual_uint8(value: Any) -> np.ndarray:
    import torch

    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
    return arr


def save_rollout_trace(
    *,
    output_dir: Path,
    eval_idx: int,
    success: bool,
    source_episode_idx: int,
    episode_seed: int,
    visuals: list[np.ndarray],
    states: list[np.ndarray],
    goal_visual: np.ndarray,
    goal_state: np.ndarray,
) -> None:
    import imageio.v2 as imageio
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "success" if success else "failure"
    stem = output_dir / f"rollout_{eval_idx:02d}_{tag}"
    states_arr = np.stack(states, axis=0).astype(np.float32)
    np.savez_compressed(
        stem.with_suffix(".npz"),
        states=states_arr,
        goal_state=np.asarray(goal_state, dtype=np.float32),
        success=np.asarray(success),
        source_episode_idx=np.asarray(source_episode_idx),
        episode_seed=np.asarray(episode_seed),
    )

    goal = normalize_visual_uint8(goal_visual)
    target_size = (goal.shape[1], goal.shape[0])
    writer = imageio.get_writer(str(stem.with_suffix(".mp4")), fps=12)
    try:
        for visual in visuals:
            frame = normalize_visual_uint8(visual)
            if frame.shape != goal.shape:
                frame = np.asarray(Image.fromarray(frame).resize(target_size, Image.BILINEAR))
            writer.append_data(np.concatenate([frame, goal], axis=1))
    finally:
        writer.close()


def to_numpy(value: Any) -> np.ndarray:
    import torch

    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def build_processors(dataset_path: Path) -> dict[str, NumpyStandardScaler]:
    processors: dict[str, NumpyStandardScaler] = {}
    with h5py.File(dataset_path, "r") as handle:
        for key in ("action", "proprio", "state"):
            if key not in handle:
                continue
            arr = np.asarray(handle[key], dtype=np.float32)
            flat = arr.reshape(arr.shape[0], -1)
            scaler = NumpyStandardScaler().fit(flat)
            processors[key] = scaler
            if key != "action":
                processors[f"goal_{key}"] = scaler
    return processors


def get_validation_episode_ids(handle: h5py.File, train_fraction: float = 0.9, seed: int = 42) -> list[int]:
    episode_ids = np.asarray(handle["ep_len"]).shape[0]
    import torch

    perm = torch.randperm(episode_ids, generator=torch.Generator().manual_seed(seed)).tolist()
    train_len = int(train_fraction * episode_ids)
    return perm[train_len:]


def get_all_episode_ids(handle: h5py.File) -> list[int]:
    return list(range(np.asarray(handle["ep_len"]).shape[0]))


def sample_valid_episode(rng: random.Random, valid_ids: list[int]) -> int:
    return valid_ids[rng.randrange(len(valid_ids))]


def episode_row_slice(handle: h5py.File, episode_idx: int) -> slice:
    start = int(handle["ep_offset"][episode_idx])
    length = int(handle["ep_len"][episode_idx])
    return slice(start, start + length)


def make_img_transform() -> Any:
    import torch
    from torchvision.transforms import v2 as transforms

    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.Resize(size=(224, 224)),
        ]
    )


def build_policy(args: argparse.Namespace, process: dict[str, Any], source_root: Path, repo_root: Path) -> tuple[Any, dict[str, Any]]:
    prepend_sys_path(source_root)
    prepend_sys_path(repo_root)

    import torch
    import stable_worldmodel as swm

    if args.policy_kind == "hf":
        model = swm.wm.utils.load_pretrained(args.policy)
    else:
        payload = torch.load(Path(args.policy).resolve(), weights_only=False, map_location="cpu")
        model = find_model_with_attr(payload, "get_cost")
        if model is None:
            model = find_model_with_attr(payload, "get_action")
        if model is None:
            raise RuntimeError(f"Could not find get_cost/get_action in {args.policy}")

    model = model.to(args.device).eval()
    model.requires_grad_(False)
    overrides = apply_model_overrides(model, args)
    img_transform = make_img_transform()

    if hasattr(model, "get_cost"):
        solver = swm.solver.CEMSolver(
            model=model,
            batch_size=1,
            num_samples=args.num_samples,
            var_scale=1.0,
            n_steps=args.n_steps,
            topk=args.topk,
            device=args.device,
            seed=args.seed,
        )
        policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=swm.PlanConfig(
                horizon=args.horizon,
                receding_horizon=args.receding_horizon,
                history_len=1,
                action_block=args.action_block,
            ),
            process=process,
            transform={"pixels": img_transform, "goal": img_transform},
        )
    else:
        policy = swm.policy.FeedForwardPolicy(
            model=model,
            process=process,
            transform={"pixels": img_transform, "goal": img_transform},
        )
    return policy, overrides


def build_wall_episode(
    handle: h5py.File,
    episode_idx: int,
    env: Any,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any], np.ndarray]:
    row = episode_row_slice(handle, episode_idx).start
    env_info = {
        "fix_door_location": np.asarray(handle["door_location"][row], dtype=np.float32),
        "fix_wall_location": np.asarray(handle["wall_location"][row], dtype=np.float32),
    }
    env.update_env(env_info)
    init_state, goal_state = env.sample_random_init_goal_states(seed)
    obs0, state0 = env.prepare(seed=seed, init_state=np.asarray(init_state, dtype=np.float32))
    obs_g, state_g = env.prepare(seed=seed, init_state=np.asarray(goal_state, dtype=np.float32))
    return obs0, np.asarray(state0, dtype=np.float32).reshape(-1), obs_g, np.asarray(state_g, dtype=np.float32).reshape(-1)


def build_pusht_episode(
    handle: h5py.File,
    episode_idx: int,
    env: Any,
    seed: int,
    frameskip: int,
    goal_h: int,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any], np.ndarray]:
    row_slice = episode_row_slice(handle, episode_idx)
    length = row_slice.stop - row_slice.start
    traj_len = frameskip * goal_h + 1
    if length < traj_len:
        raise ValueError(f"Episode {episode_idx} too short for traj_len={traj_len}: {length}")
    max_offset = length - traj_len
    rng = random.Random(seed)
    offset = rng.randint(0, max_offset)
    start = row_slice.start + offset

    if "shape" in handle:
        shape_arr = handle["shape"][start]
        shape = shape_arr.decode() if isinstance(shape_arr, bytes) else str(shape_arr)
    else:
        # Official DINO-WM PushT loader defaults to the original T-shape when
        # shapes.pkl is absent. Keep eval compatible with previously converted HDF5s.
        shape = "T"
    env.update_env({"shape": shape})

    init_state = np.asarray(handle["state"][start], dtype=np.float32)
    exec_actions = np.asarray(handle["action"][start : start + frameskip * goal_h], dtype=np.float32)
    rollout_obses, rollout_states = env.rollout(seed, init_state, exec_actions)
    obs0 = {
        "visual": rollout_obses["visual"][0],
        "proprio": rollout_obses["proprio"][0],
    }
    obs_g = {
        "visual": rollout_obses["visual"][-1],
        "proprio": rollout_obses["proprio"][-1],
    }
    state0 = np.asarray(rollout_states[0], dtype=np.float32).reshape(-1)
    state_g = np.asarray(rollout_states[-1], dtype=np.float32).reshape(-1)
    return obs0, state0, obs_g, state_g


def build_policy_inputs(
    obs: dict[str, Any],
    state: np.ndarray,
    goal_obs: dict[str, Any],
    goal_state: np.ndarray,
) -> dict[str, Any]:
    pixels = normalize_visual_uint8(obs["visual"])[None, None, ...]
    goal = normalize_visual_uint8(goal_obs["visual"])[None, None, ...]
    proprio = to_numpy(obs["proprio"]).reshape(1, 1, -1).astype(np.float32)
    goal_proprio = to_numpy(goal_obs["proprio"]).reshape(1, 1, -1).astype(np.float32)
    state = np.asarray(state, dtype=np.float32).reshape(1, 1, -1)
    goal_state = np.asarray(goal_state, dtype=np.float32).reshape(1, 1, -1)
    return {
        "pixels": pixels,
        "goal": goal,
        "proprio": proprio,
        "goal_proprio": goal_proprio,
        "state": state,
        "goal_state": goal_state,
    }


def step_env(env: Any, action: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    out = env.step(action)
    if len(out) != 4:
        raise ValueError(f"Unexpected env.step return length: {len(out)}")
    obs, _, _, info = out
    return obs, info


def load_env(spec: dict[str, Any], source_root: Path) -> Any:
    install_d4rl_stub()
    prepend_sys_path(source_root)
    module_name, class_name = spec["env_ctor"]
    module = __import__(module_name, fromlist=[class_name])
    ctor = getattr(module, class_name)
    if spec["env_ctor"][1] == "WallEnvWrapper":
        return ctor(device="cpu")
    return ctor(with_velocity=True, with_target=True)


def write_report(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        handle.write("==== CONFIG ====\n")
        handle.write(json.dumps(payload, indent=2))
        handle.write("\n\n==== RESULTS ====\n")
        handle.write(f"success_rate: {payload['success_rate']}\n")
        for row in payload["episodes"]:
            handle.write(
                f"episode={row['eval_idx']} seed={row['episode_seed']} "
                f"success={row['success']} steps={row['steps']} "
                f"benchmark_episode={row['source_episode_idx']}\n"
            )
        handle.write(f"evaluation_time: {payload['elapsed_sec']} seconds\n")


def load_eval_plan(plan_path: Path) -> list[dict[str, int]]:
    payload = json.loads(plan_path.read_text())
    if isinstance(payload, dict):
        episodes = payload.get("episodes")
    else:
        episodes = payload
    if not isinstance(episodes, list):
        raise ValueError(f"Eval plan at {plan_path} must contain an episodes list")

    normalized: list[dict[str, int]] = []
    for idx, row in enumerate(episodes):
        if not isinstance(row, dict):
            raise ValueError(f"Eval plan row {idx} is not an object")
        if "source_episode_idx" not in row or "episode_seed" not in row:
            raise ValueError(
                f"Eval plan row {idx} must contain source_episode_idx and episode_seed"
            )
        normalized.append(
            {
                "source_episode_idx": int(row["source_episode_idx"]),
                "episode_seed": int(row["episode_seed"]),
            }
        )
    return normalized


def build_payload(
    *,
    args: argparse.Namespace,
    overrides: dict[str, Any],
    dataset_path: Path,
    val_dataset_path: Path,
    rows: list[dict[str, Any]],
    successes: list[float],
    start_time: float,
) -> dict[str, Any]:
    return {
        "benchmark": args.benchmark,
        "policy_kind": args.policy_kind,
        "policy": args.policy,
        "eval_plan": str(args.eval_plan.resolve()) if args.eval_plan is not None else None,
        "dataset_path": str(dataset_path),
        "val_dataset_path": str(val_dataset_path),
        "seed": args.seed,
        "n_eval": args.n_eval,
        "num_samples": args.num_samples,
        "n_steps": args.n_steps,
        "topk": args.topk,
        "horizon": args.horizon,
        "receding_horizon": args.receding_horizon,
        "action_block": args.action_block,
        "model_overrides": overrides,
        "episodes": rows,
        "success_rate": float(np.mean(successes) * 100.0) if successes else 0.0,
        "elapsed_sec": time.time() - start_time,
    }


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    source_root = (
        args.source_root.resolve()
        if args.source_root is not None
        else (repo_root / "external" / "dino_wm").resolve()
    )
    spec = BENCHMARK_SPECS[args.benchmark]
    dataset_path = (
        args.dataset_path.resolve()
        if args.dataset_path is not None
        else (repo_root / spec["default_dataset"]).resolve()
    )
    val_dataset_path = (
        args.val_dataset_path.resolve()
        if args.val_dataset_path is not None
        else (
            (repo_root / "data" / "dino_pusht_noise_val.h5").resolve()
            if args.benchmark == "pusht_noise"
            else dataset_path
        )
    )
    args.action_block = args.action_block or spec["action_block"]

    swm_source = repo_root / "external" / "stable-worldmodel"
    prepend_sys_path(swm_source)
    policy, overrides = build_policy(
        args=args,
        process=build_processors(dataset_path),
        source_root=swm_source,
        repo_root=repo_root,
    )
    env = load_env(spec, source_root)
    policy.set_env(_PolicyEnvAdapter(make_batched_action_space(get_or_create_action_space(env))))

    start_time = time.time()
    rows: list[dict[str, Any]] = []
    successes: list[float] = []

    with h5py.File(val_dataset_path, "r") as handle:
        eval_plan = load_eval_plan(args.eval_plan.resolve()) if args.eval_plan is not None else None
        if eval_plan is None:
            if args.benchmark == "pusht_noise" and val_dataset_path != dataset_path:
                valid_ids = get_all_episode_ids(handle)
            else:
                valid_ids = get_validation_episode_ids(handle)
            rng = random.Random(args.seed)
            eval_rows = [
                {
                    "source_episode_idx": int(sample_valid_episode(rng, valid_ids)),
                    "episode_seed": int(args.seed * eval_idx + 1),
                }
                for eval_idx in range(args.n_eval)
            ]
        else:
            eval_rows = eval_plan
            args.n_eval = len(eval_rows)

        for eval_idx, eval_row in enumerate(eval_rows):
            episode_seed = int(eval_row["episode_seed"])
            source_episode_idx = int(eval_row["source_episode_idx"])
            if args.benchmark == "wall":
                obs0, state0, obs_g, state_g = build_wall_episode(
                    handle, source_episode_idx, env, episode_seed
                )
            else:
                obs0, state0, obs_g, state_g = build_pusht_episode(
                    handle,
                    source_episode_idx,
                    env,
                    episode_seed,
                    spec["frameskip"],
                    spec["goal_h"],
                )

            # Episode builders may touch the live env to render the goal observation.
            # Restore the actual rollout start state before planning/execution.
            obs0, state0 = env.prepare(
                seed=episode_seed,
                init_state=np.asarray(state0, dtype=np.float32),
            )

            reset_policy_state(policy)
            obs = obs0
            cur_state = np.asarray(state0, dtype=np.float32)
            success = False
            steps = 0
            rollout_visuals = [normalize_visual_uint8(obs["visual"])]
            rollout_states = [cur_state.copy()]

            for step in range(spec["max_episode_steps"]):
                info_dict = build_policy_inputs(obs, cur_state, obs_g, state_g)
                action = np.asarray(policy.get_action(info_dict), dtype=np.float32)
                if action.ndim == 2:
                    action = action[0]
                obs, env_info = step_env(env, action)
                cur_state = np.asarray(env_info["state"], dtype=np.float32).reshape(-1)
                rollout_visuals.append(normalize_visual_uint8(obs["visual"]))
                rollout_states.append(cur_state.copy())
                success = bool(
                    env.eval_state(
                        np.asarray(state_g, dtype=np.float32).reshape(-1),
                        np.asarray(cur_state, dtype=np.float32).reshape(-1),
                    )["success"]
                )
                steps = step + 1
                if success:
                    break

            rows.append(
                {
                    "eval_idx": eval_idx,
                    "episode_seed": episode_seed,
                    "source_episode_idx": int(source_episode_idx),
                    "success": float(success),
                    "steps": steps,
                }
            )
            successes.append(float(success))
            if args.save_rollout_dir is not None:
                save_rollout_trace(
                    output_dir=args.save_rollout_dir,
                    eval_idx=eval_idx,
                    success=success,
                    source_episode_idx=source_episode_idx,
                    episode_seed=episode_seed,
                    visuals=rollout_visuals,
                    states=rollout_states,
                    goal_visual=obs_g["visual"],
                    goal_state=state_g,
                )
            write_report(
                args.output,
                build_payload(
                    args=args,
                    overrides=overrides,
                    dataset_path=dataset_path,
                    val_dataset_path=val_dataset_path,
                    rows=rows,
                    successes=successes,
                    start_time=start_time,
                ),
            )

    payload = build_payload(
        args=args,
        overrides=overrides,
        dataset_path=dataset_path,
        val_dataset_path=val_dataset_path,
        rows=rows,
        successes=successes,
        start_time=start_time,
    )
    write_report(args.output, payload)


if __name__ == "__main__":
    main()
