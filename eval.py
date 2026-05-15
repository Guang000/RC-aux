import os
import json

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


def load_row_indices(path_str):
    path = Path(path_str).expanduser()
    text = path.read_text()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, list):
        return np.asarray(data, dtype=np.int64)
    if isinstance(data, dict) and "row_indices" in data:
        return np.asarray(data["row_indices"], dtype=np.int64)

    cfg_match = None
    if "==== CONFIG ====" in text and "==== RESULTS ====" in text:
        cfg_match = text.split("==== CONFIG ====", 1)[1].split("==== RESULTS ====", 1)[0]
    if cfg_match is not None:
        cfg_match = cfg_match.strip()
        try:
            data = json.loads(cfg_match)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and "row_indices" in data:
            return np.asarray(data["row_indices"], dtype=np.int64)

    raise ValueError(f"Could not load row_indices from: {path}")

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.get("cache_dir") or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    cache_dir = Path(cfg.get("cache_dir") or swm.data.utils.get_cache_dir())

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    policy = cfg.get("policy", "random")

    if policy != "random":
        model = swm.policy.AutoCostModel(cfg.policy, cache_dir=cache_dir)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        planner_override = cfg.get("planner_override")
        if planner_override is not None:
            if "use_reachability_cost" in planner_override:
                model.use_reachability_cost = bool(
                    planner_override.use_reachability_cost
                )
            if "reachability_cost_weight" in planner_override:
                model.reachability_cost_weight = float(
                    planner_override.reachability_cost_weight
                )
            if "latent_cost_weight" in planner_override:
                model.latent_cost_weight = float(planner_override.latent_cost_weight)
            if "goal_cost_reduce" in planner_override:
                model.goal_cost_reduce = str(planner_override.goal_cost_reduce)
            if "goal_cost_softmin_temperature" in planner_override:
                model.goal_cost_softmin_temperature = float(
                    planner_override.goal_cost_softmin_temperature
                )
            if "use_temporal_distance_cost" in planner_override:
                model.use_temporal_distance_cost = bool(
                    planner_override.use_temporal_distance_cost
                )
            if "temporal_distance_cost_weight" in planner_override:
                model.temporal_distance_cost_weight = float(
                    planner_override.temporal_distance_cost_weight
                )
            if "temporal_distance_cost_reduce" in planner_override:
                model.temporal_distance_cost_reduce = str(
                    planner_override.temporal_distance_cost_reduce
                )
            if "action_l2_cost_weight" in planner_override:
                model.action_l2_cost_weight = float(
                    planner_override.action_l2_cost_weight
                )
            if "action_smooth_cost_weight" in planner_override:
                model.action_smooth_cost_weight = float(
                    planner_override.action_smooth_cost_weight
                )
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(cache_dir, cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    fixed_row_indices_file = cfg.eval.get("row_indices_file")
    fixed_row_indices = cfg.eval.get("row_indices")
    if fixed_row_indices_file is not None or fixed_row_indices is not None:
        if fixed_row_indices_file is not None:
            random_episode_indices = load_row_indices(fixed_row_indices_file)
        else:
            random_episode_indices = np.asarray(fixed_row_indices, dtype=np.int64)
        random_episode_indices = np.sort(random_episode_indices)
        if len(random_episode_indices) != cfg.eval.num_eval:
            raise ValueError(
                "eval.row_indices must contain exactly eval.num_eval rows."
            )
        if not np.isin(random_episode_indices, valid_indices).all():
            raise ValueError(
                "Some provided eval.row_indices are not valid starting points for the current protocol."
            )
    else:
        g = np.random.default_rng(cfg.seed)
        legacy_sample_indices = os.getenv("LEWM_LEGACY_SAMPLE_INDICES") == "1"
        if legacy_sample_indices:
            sampled = g.choice(
                len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
            )
            random_episode_indices = np.sort(valid_indices[sampled])
        else:
            random_episode_indices = g.choice(
                valid_indices, size=cfg.eval.num_eval, replace=False
            )
            random_episode_indices = np.sort(random_episode_indices)

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        save_video=cfg.eval.get("save_video", False),
        video_path=results_path,
    )
    end_time = time.time()
    
    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
