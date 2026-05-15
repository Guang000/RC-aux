import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import (
    ARPredictor,
    AdaptiveSIGReg,
    Embedder,
    GroundingHead,
    MLP,
    ReachabilityHead,
    SIGReg,
    TemporalDistanceHead,
)
from rcaux import maybe_expand_num_preds, maybe_freeze_for_reachability, rcaux_forward
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


def lejepa_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, and compute LeWM losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    reg_type = cfg.loss.get("reg_type", "sigreg")
    reg_cfg = cfg.loss.get(reg_type)
    lambd = reg_cfg.weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def build_regularizer(cfg):
    reg_type = cfg.loss.get("reg_type", "sigreg")
    regularizers = {
        "sigreg": SIGReg,
        "adaptive_sigreg": AdaptiveSIGReg,
    }
    if reg_type not in regularizers:
        raise ValueError(f"Unknown regularizer type: {reg_type}")
    return regularizers[reg_type](**cfg.loss[reg_type].kwargs)


def maybe_load_init_weights(model, cfg):
    init_cfg = cfg.get("init")
    init_path = init_cfg.get("weights_path") if init_cfg is not None else None
    if not init_path:
        return

    ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, torch.nn.Module):
        state_dict = ckpt.state_dict()
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = {
            k.removeprefix("model."): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise TypeError(f"Unsupported init checkpoint type: {type(ckpt)}")

    strict = bool(init_cfg.get("strict", True))
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    print(
        f"Loaded init weights from {init_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    if cfg.get("seed") is not None:
        pl.seed_everything(int(cfg.seed), workers=True)

    use_rcaux = (
        cfg.loss.get("prediction_type", "one_step") != "one_step"
        or cfg.loss.get("reachability", {}).get("enabled", False)
    )
    if use_rcaux:
        maybe_expand_num_preds(cfg)

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    reach_cfg = cfg.loss.get("reachability")
    reachability_head = None
    if reach_cfg is not None and reach_cfg.enabled:
        reachability_head = ReachabilityHead(
            embed_dim=embed_dim,
            hidden_dim=reach_cfg.hidden_dim,
            max_horizon=reach_cfg.max_horizon,
            horizon_dim=reach_cfg.horizon_dim,
        )

    ground_cfg = cfg.loss.get("grounding")
    grounding_head = None
    if ground_cfg is not None and ground_cfg.enabled:
        target_key = ground_cfg.target_key
        output_dim = int(getattr(cfg.wm, f"{target_key}_dim"))
        grounding_head = GroundingHead(
            embed_dim=embed_dim,
            output_dim=output_dim,
            hidden_dim=ground_cfg.hidden_dim,
        )

    td_cfg = cfg.loss.get("temporal_distance")
    temporal_distance_head = None
    if td_cfg is not None and td_cfg.enabled:
        temporal_distance_head = TemporalDistanceHead(
            embed_dim=embed_dim,
            hidden_dim=td_cfg.hidden_dim,
            max_horizon=td_cfg.max_horizon,
        )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
        reachability_head=reachability_head,
        grounding_head=grounding_head,
        temporal_distance_head=temporal_distance_head,
        use_reachability_cost=bool(
            reach_cfg is not None
            and reach_cfg.enabled
        ),
        reachability_cost_weight=float(reach_cfg.get("planner_weight", 0.0))
        if reach_cfg is not None
        else 0.0,
        latent_cost_weight=float(reach_cfg.get("latent_weight", 1.0))
        if reach_cfg is not None
        else 1.0,
        goal_cost_reduce=cfg.loss.get("planner", {}).get("goal_cost_reduce", "terminal"),
        goal_cost_softmin_temperature=float(
            cfg.loss.get("planner", {}).get("goal_cost_softmin_temperature", 1.0)
        ),
        action_l2_cost_weight=float(
            cfg.loss.get("planner", {}).get("action_l2_cost_weight", 0.0)
        ),
        action_smooth_cost_weight=float(
            cfg.loss.get("planner", {}).get("action_smooth_cost_weight", 0.0)
        ),
    )
    maybe_load_init_weights(world_model, cfg)
    if use_rcaux and reach_cfg is not None:
        maybe_freeze_for_reachability(world_model, cfg)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = build_regularizer(cfg),
        forward=partial(rcaux_forward if use_rcaux else lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    num_sanity_val_steps = trainer_kwargs.pop("num_sanity_val_steps", 1)

    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=num_sanity_val_steps,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        seed=cfg.seed,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
