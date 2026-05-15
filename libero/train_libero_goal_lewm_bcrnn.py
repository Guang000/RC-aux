#!/usr/bin/env python3
"""Train a sequential LIBERO-Goal action head on a LeWM-family encoder."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import tv_tensors


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "libero"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_libero_goal_lewm_bc import (  # noqa: E402
    DEFAULT_LEWM,
    LIBERO_ROOT,
    LEWM_CODE,
    SafeScaler,
    demo_eval_lowdim,
    demo_sort_key,
    load_world_model,
    make_image_transform,
    prepend,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-root", type=Path, default=LIBERO_ROOT)
    parser.add_argument("--init-policy", type=Path, default=DEFAULT_LEWM)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "runs" / "libero_goal_lewm_bcrnn",
    )
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument(
        "--tasks",
        default="all",
        help="LIBERO-Goal task ids to train on, e.g. 'all', '7', or '0,1,2'.",
    )
    parser.add_argument("--image-keys", default="agentview_rgb")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--train-encoder", action="store_true")
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume from a LeWM-BCRNN checkpoint produced by this script.",
    )
    parser.add_argument(
        "--resume-training-state",
        type=Path,
        default=None,
        help="Optional training_state.pt with optimizer/history for resume.",
    )
    return parser.parse_args()


def parse_image_keys(value: str) -> list[str]:
    keys = [part.strip() for part in value.split(",") if part.strip()]
    if not keys:
        raise ValueError("--image-keys must contain at least one observation key")
    return keys


def parse_task_ids(value: str, n_tasks: int) -> list[int]:
    if value.lower() == "all":
        return list(range(n_tasks))
    task_ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not task_ids:
        raise ValueError("--tasks must be 'all' or contain at least one task id")
    bad = [task_id for task_id in task_ids if task_id < 0 or task_id >= n_tasks]
    if bad:
        raise ValueError(f"Task ids out of range for LIBERO-Goal: {bad}")
    return task_ids


class LiberoGoalLeWMSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        libero_root: Path,
        transform: Any,
        img_size: int,
        seq_len: int,
        image_keys: list[str],
        task_ids: list[int] | None = None,
    ):
        self.libero_root = libero_root
        self.transform = transform
        self.img_size = img_size
        self.seq_len = seq_len
        self.image_keys = image_keys
        self.task_ids = task_ids
        self.files: list[Path] = []
        self.samples: list[tuple[int, int, str, int]] = []
        self._handles: dict[int, h5py.File] = {}

        prepend(libero_root)
        from libero.libero import benchmark  # noqa: WPS433

        bench = benchmark.get_benchmark_dict()["libero_goal"]()
        if self.task_ids is None:
            self.task_ids = list(range(bench.n_tasks))
        for task_id in self.task_ids:
            path = libero_root / "datasets" / bench.get_task_demonstration(task_id)
            file_idx = len(self.files)
            self.files.append(path)
            with h5py.File(path, "r") as handle:
                demos = handle["data"]
                for demo_name in sorted(demos.keys(), key=demo_sort_key):
                    length = int(demos[demo_name]["actions"].shape[0])
                    first = max(0, self.seq_len - 1)
                    for step in range(first, length):
                        self.samples.append((task_id, file_idx, demo_name, step))

        actions, proprios = [], []
        for path in self.files:
            with h5py.File(path, "r") as handle:
                for demo_name in sorted(handle["data"].keys(), key=demo_sort_key):
                    demo = handle["data"][demo_name]
                    actions.append(np.asarray(demo["actions"], dtype=np.float32))
                    proprios.append(demo_eval_lowdim(demo, slice(None)))
        action_arr = np.concatenate(actions, axis=0)
        proprio_arr = np.concatenate(proprios, axis=0)
        self.action_scaler = SafeScaler(action_arr.mean(0, keepdims=True), action_arr.std(0, keepdims=True))
        self.proprio_scaler = SafeScaler(proprio_arr.mean(0, keepdims=True), proprio_arr.std(0, keepdims=True))

    def __len__(self) -> int:
        return len(self.samples)

    def _handle(self, file_idx: int) -> h5py.File:
        if file_idx not in self._handles:
            self._handles[file_idx] = h5py.File(self.files[file_idx], "r")
        return self._handles[file_idx]

    def _prep_image(self, image: np.ndarray) -> torch.Tensor:
        if image.shape[0] != self.img_size or image.shape[1] != self.img_size:
            image = np.asarray(Image.fromarray(image).resize((self.img_size, self.img_size), Image.BILINEAR))
        image = np.transpose(image.astype(np.uint8, copy=False), (2, 0, 1))
        return self.transform(tv_tensors.Image(image))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        task_id, file_idx, demo_name, end_step = self.samples[index]
        demo = self._handle(file_idx)["data"][demo_name]
        start = end_step - self.seq_len + 1
        indices = list(range(start, end_step + 1))
        pixels = torch.stack(
            [
                torch.stack([self._prep_image(np.asarray(demo["obs"][key][i])) for key in self.image_keys], dim=0)
                for i in indices
            ],
            dim=0,
        )
        actions = self.action_scaler.transform(np.asarray(demo["actions"][indices], dtype=np.float32))
        proprios = self.proprio_scaler.transform(demo_eval_lowdim(demo, indices))
        task = np.zeros((10,), dtype=np.float32)
        task[task_id] = 1.0
        task_seq = np.repeat(task[None, :], self.seq_len, axis=0)
        return {
            "pixels": pixels,
            "actions": torch.from_numpy(actions.astype(np.float32)),
            "proprios": torch.from_numpy(proprios.astype(np.float32)),
            "tasks": torch.from_numpy(task_seq.astype(np.float32)),
        }


class LeWMLiberoBCRNNHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        proprio_dim: int,
        task_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.in_norm = nn.LayerNorm(embed_dim + proprio_dim + task_dim)
        self.rnn = nn.LSTM(
            input_size=embed_dim + proprio_dim + task_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, emb: torch.Tensor, proprio: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emb.float(), proprio.float(), task.float()], dim=-1)
        x = self.in_norm(x)
        y, _ = self.rnn(x)
        return self.out(y)


class LeWMLiberoBCRNNPolicy(nn.Module):
    def __init__(self, world_model: nn.Module, head: LeWMLiberoBCRNNHead):
        super().__init__()
        self.world_model = world_model
        self.head = head

    def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim == 5:
            return self.world_model.encode({"pixels": pixels})["emb"]
        if pixels.ndim != 6:
            raise ValueError(f"Expected pixels with 5 or 6 dims, got {tuple(pixels.shape)}")
        batch, seq_len, n_views, channels, height, width = pixels.shape
        flat = pixels.permute(0, 2, 1, 3, 4, 5).reshape(batch * n_views, seq_len, channels, height, width)
        emb = self.world_model.encode({"pixels": flat})["emb"]
        return emb.reshape(batch, n_views, seq_len, emb.shape[-1]).permute(0, 2, 1, 3).reshape(batch, seq_len, -1)

    def forward(self, pixels: torch.Tensor, proprio: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        emb = self.encode_pixels(pixels)
        return self.head(emb, proprio, task)


def save_checkpoint(
    policy: LeWMLiberoBCRNNPolicy,
    optimizer: torch.optim.Optimizer,
    run_dir: Path,
    epoch: int,
    args: argparse.Namespace,
    history: list[dict[str, float]],
    head_config: dict[str, Any],
    dataset: LiberoGoalLeWMSequenceDataset,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / f"lewm_libero_bcrnn_epoch_{epoch}.ckpt"
    policy_cpu = policy.cpu()
    torch.save(
        {
            "world_model": policy_cpu.world_model,
            "head_state_dict": policy_cpu.head.state_dict(),
            "head_config": head_config,
            "action_mean": dataset.action_scaler.mean,
            "action_std": dataset.action_scaler.std,
            "proprio_mean": dataset.proprio_scaler.mean,
            "proprio_std": dataset.proprio_scaler.std,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith("_")},
            "history": history,
        },
        model_path,
    )
    policy.to(args._device)
    torch.save(
        {
            "epoch": epoch,
            "head_state_dict": policy.head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "head_config": head_config,
        },
        run_dir / "training_state.pt",
    )
    (run_dir / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
    (run_dir / "config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith("_")}, indent=2) + "\n"
    )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(args.libero_root / ".libero"))
    prepend(args.libero_root)
    prepend(LEWM_CODE)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("medium")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args._device = device
    transform = make_image_transform(args.img_size)
    image_keys = parse_image_keys(args.image_keys)
    prepend(args.libero_root)
    from libero.libero import benchmark  # noqa: WPS433

    bench = benchmark.get_benchmark_dict()["libero_goal"]()
    task_ids = parse_task_ids(args.tasks, bench.n_tasks)
    dataset = LiberoGoalLeWMSequenceDataset(args.libero_root, transform, args.img_size, args.seq_len, image_keys, task_ids)
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        generator=generator,
    )

    world_model = load_world_model(args.init_policy, device)
    world_model.train(args.train_encoder)
    world_model.requires_grad_(args.train_encoder)
    base_embed_dim = int(world_model.predictor.pos_embedding.shape[-1])
    embed_dim = base_embed_dim * len(image_keys)
    head_config = {
        "embed_dim": embed_dim,
        "proprio_dim": 9,
        "task_dim": 10,
        "action_dim": 7,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "seq_len": args.seq_len,
        "image_keys": image_keys,
        "task_ids": task_ids,
    }
    head = LeWMLiberoBCRNNHead(**{k: v for k, v in head_config.items() if k not in {"seq_len", "image_keys", "task_ids"}})
    policy = LeWMLiberoBCRNNPolicy(world_model, head).to(device)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    use_amp = device.type == "cuda" and args.precision == "bf16"

    history: list[dict[str, float]] = []
    start_epoch = 1
    if args.resume_checkpoint is not None:
        payload = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        policy.world_model = payload["world_model"].to(device)
        policy.world_model.train(args.train_encoder)
        policy.world_model.requires_grad_(args.train_encoder)
        policy.head.load_state_dict(payload["head_state_dict"])
        policy.to(device)
        trainable = [p for p in policy.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
        state_path = args.resume_training_state or args.resume_checkpoint.parent / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            if "optimizer_state_dict" in state:
                optimizer.load_state_dict(state["optimizer_state_dict"])
            history = list(state.get("history", payload.get("history", [])))
            start_epoch = int(state.get("epoch", len(history))) + 1
        else:
            history = list(payload.get("history", []))
            start_epoch = int(history[-1]["epoch"]) + 1 if history else 1

    print(
        json.dumps(
            {
                "dataset_len": len(dataset),
                "seq_len": args.seq_len,
                "task_ids": task_ids,
                "image_keys": image_keys,
                "init_policy": str(args.init_policy),
                "run_dir": str(args.run_dir),
                "device": str(device),
                "train_encoder": bool(args.train_encoder),
                "trainable_parameters": int(sum(p.numel() for p in trainable)),
                "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
                "start_epoch": start_epoch,
                "max_epochs": args.max_epochs,
            },
            indent=2,
        ),
        flush=True,
    )

    global_start = time.time()
    for epoch in range(start_epoch, args.max_epochs + 1):
        policy.head.train()
        policy.world_model.train(args.train_encoder)
        totals = {"loss": 0.0, "mae": 0.0, "last_mae": 0.0}
        num_batches = 0
        epoch_start = time.time()
        for batch_idx, batch in enumerate(loader, start=1):
            pixels = batch["pixels"].to(device, non_blocking=True)
            proprios = batch["proprios"].to(device, non_blocking=True)
            tasks = batch["tasks"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                if args.train_encoder:
                    pred = policy(pixels, proprios, tasks)
                else:
                    with torch.no_grad():
                        emb = policy.encode_pixels(pixels).float()
                    pred = policy.head(emb, proprios, tasks)
                loss = torch.nn.functional.smooth_l1_loss(pred.float(), actions.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            with torch.no_grad():
                mae = (pred.float() - actions.float()).abs().mean()
                last_mae = (pred[:, -1].float() - actions[:, -1].float()).abs().mean()
            totals["loss"] += float(loss.detach().cpu())
            totals["mae"] += float(mae.detach().cpu())
            totals["last_mae"] += float(last_mae.detach().cpu())
            num_batches += 1
            if batch_idx % 50 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "batch": batch_idx,
                            "loss": totals["loss"] / num_batches,
                            "mae": totals["mae"] / num_batches,
                            "last_mae": totals["last_mae"] / num_batches,
                            "elapsed_sec": time.time() - global_start,
                        }
                    ),
                    flush=True,
                )
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
        row = {
            "epoch": epoch,
            "batches": num_batches,
            "loss": totals["loss"] / max(num_batches, 1),
            "mae": totals["mae"] / max(num_batches, 1),
            "last_mae": totals["last_mae"] / max(num_batches, 1),
            "epoch_sec": time.time() - epoch_start,
            "elapsed_sec": time.time() - global_start,
        }
        history.append(row)
        print(json.dumps(row, indent=2), flush=True)
        if epoch % args.save_every == 0 or epoch == args.max_epochs:
            save_checkpoint(policy, optimizer, args.run_dir, epoch, args, history, head_config, dataset)


if __name__ == "__main__":
    main()
