from omegaconf import open_dict
import torch


def maybe_expand_num_preds(cfg):
    """Ensure the dataset is long enough to activate RC-aux horizons."""

    pred_type = cfg.loss.get("prediction_type", "one_step")
    reach_cfg = cfg.loss.get("reachability")
    current = int(cfg.wm.num_preds)
    required = current

    if pred_type == "multi_horizon":
        required = max(
            required,
            int(cfg.loss.rollout.max_horizon),
            int(cfg.loss.rollout.get("prefix_horizon", 0)),
            1,
        )

    if reach_cfg is not None and reach_cfg.enabled:
        required = max(required, int(reach_cfg.max_horizon))

    if required != current:
        print(
            "Expanding wm.num_preds for RC-aux "
            f"(current={current}, required={required})."
        )

    with open_dict(cfg):
        cfg.wm.num_preds = required
        cfg.data.dataset.num_steps = cfg.wm.history_size + cfg.wm.num_preds


def get_rollout_horizon(module, cfg, stage, available_horizon):
    rollout_cfg = cfg.loss.rollout
    max_horizon = min(int(rollout_cfg.max_horizon), int(available_horizon))
    min_horizon = min(int(rollout_cfg.min_horizon), max_horizon)

    if rollout_cfg.curriculum_epochs > 0:
        progress = min(
            1.0,
            float(module.current_epoch + 1) / float(rollout_cfg.curriculum_epochs),
        )
        scheduled = min_horizon + int(round((max_horizon - min_horizon) * progress))
        max_horizon = max(min_horizon, min(max_horizon, scheduled))

    if stage == "fit" and rollout_cfg.sample and max_horizon > min_horizon:
        return int(torch.randint(min_horizon, max_horizon + 1, ()).item())

    return max_horizon


def get_rollout_weights(horizon, cfg, device, dtype):
    weighting = cfg.loss.rollout.weighting
    if weighting == "uniform":
        weights = torch.ones(horizon, device=device, dtype=dtype)
    elif weighting == "linear":
        weights = torch.arange(1, horizon + 1, device=device, dtype=dtype)
    elif weighting == "power":
        weights = torch.arange(1, horizon + 1, device=device, dtype=dtype).pow(
            cfg.loss.rollout.weight_power
        )
    else:
        raise ValueError(f"Unknown rollout weighting: {weighting}")

    return weights / weights.sum().clamp_min(torch.finfo(dtype).eps)


def one_step_prediction_loss(model, emb, act_emb, ctx_len):
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, 1 : ctx_len + 1]
    pred_emb = model.predict(ctx_emb, ctx_act)
    step_losses = (pred_emb - tgt_emb).pow(2).mean(dim=(0, 2))
    return step_losses.mean(), pred_emb, tgt_emb, step_losses


def open_loop_prediction_loss(module, emb, act_emb, cfg, rollout_horizon):
    ctx_len = cfg.wm.history_size
    init_emb = emb[:, :ctx_len]
    init_act = act_emb[:, :ctx_len]
    future_act = act_emb[:, ctx_len : ctx_len + rollout_horizon - 1]
    tgt_emb = emb[:, ctx_len : ctx_len + rollout_horizon]

    pred_emb = module.model.rollout_open_loop(
        init_emb,
        init_act,
        future_act,
        horizon=rollout_horizon,
        history_size=ctx_len,
    )

    step_losses = (pred_emb - tgt_emb).pow(2).mean(dim=(0, 2))
    weights = get_rollout_weights(
        rollout_horizon,
        cfg,
        device=step_losses.device,
        dtype=step_losses.dtype,
    )
    pred_loss = (step_losses * weights).sum()
    return pred_loss, pred_emb, tgt_emb, step_losses


def multi_horizon_prediction_loss(module, emb, act_emb, cfg, stage):
    ctx_len = cfg.wm.history_size
    available_horizon = emb.size(1) - ctx_len
    rollout_horizon = get_rollout_horizon(module, cfg, stage, available_horizon)
    pred_loss, pred_emb, tgt_emb, step_losses = open_loop_prediction_loss(
        module, emb, act_emb, cfg, rollout_horizon
    )
    return pred_loss, pred_emb, tgt_emb, step_losses, rollout_horizon


def _expand_reach_pairs(src, future_seq):
    horizons = future_seq.size(1)
    src_chunks = []
    goal_chunks = []
    budget_chunks = []
    delta_chunks = []
    for delta in range(1, horizons + 1):
        budgets = torch.arange(delta, horizons + 1, device=src.device, dtype=torch.long)
        num_budget = budgets.numel()
        src_chunks.append(src.unsqueeze(1).expand(-1, num_budget, -1))
        goal_chunks.append(future_seq[:, delta - 1 : delta].expand(-1, num_budget, -1))
        budget_chunks.append(budgets.unsqueeze(0).expand(src.size(0), -1))
        delta_chunks.append(
            torch.full(
                (src.size(0), num_budget),
                delta,
                device=src.device,
                dtype=torch.long,
            )
        )

    return (
        torch.cat(src_chunks, dim=1),
        torch.cat(goal_chunks, dim=1),
        torch.cat(budget_chunks, dim=1),
        torch.cat(delta_chunks, dim=1),
    )


def _expand_temporal_negative_pairs(src, future_seq):
    horizons = future_seq.size(1)
    src_chunks = []
    goal_chunks = []
    budget_chunks = []
    delta_chunks = []
    for delta in range(2, horizons + 1):
        budgets = torch.arange(1, delta, device=src.device, dtype=torch.long)
        num_budget = budgets.numel()
        src_chunks.append(src.unsqueeze(1).expand(-1, num_budget, -1))
        goal_chunks.append(future_seq[:, delta - 1 : delta].expand(-1, num_budget, -1))
        budget_chunks.append(budgets.unsqueeze(0).expand(src.size(0), -1))
        delta_chunks.append(
            torch.full(
                (src.size(0), num_budget),
                delta,
                device=src.device,
                dtype=torch.long,
            )
        )

    if not src_chunks:
        empty_feat = src.new_empty(src.size(0), 0, src.size(-1))
        empty_budget = torch.empty(src.size(0), 0, device=src.device, dtype=torch.long)
        return empty_feat, empty_feat, empty_budget, empty_budget

    return (
        torch.cat(src_chunks, dim=1),
        torch.cat(goal_chunks, dim=1),
        torch.cat(budget_chunks, dim=1),
        torch.cat(delta_chunks, dim=1),
    )


def _expand_temporal_distance_pairs(src, future_seq):
    horizons = future_seq.size(1)
    src_chunks = []
    goal_chunks = []
    delta_chunks = []
    for delta in range(1, horizons + 1):
        src_chunks.append(src)
        goal_chunks.append(future_seq[:, delta - 1])
        delta_chunks.append(
            torch.full(
                (src.size(0),),
                delta,
                device=src.device,
                dtype=torch.long,
            )
        )

    return (
        torch.cat(src_chunks, dim=0),
        torch.cat(goal_chunks, dim=0),
        torch.cat(delta_chunks, dim=0),
    )


def _collect_rollout_pairs(sequence, expand_fn):
    src_chunks = []
    goal_chunks = []
    budget_chunks = []
    delta_chunks = []

    horizons = sequence.size(1) - 1
    for src_idx in range(horizons):
        src_i, goal_i, budget_i, delta_i = expand_fn(
            sequence[:, src_idx],
            sequence[:, src_idx + 1 :],
        )
        if budget_i.numel() == 0:
            continue
        src_chunks.append(src_i)
        goal_chunks.append(goal_i)
        budget_chunks.append(budget_i)
        delta_chunks.append(delta_i)

    if not src_chunks:
        batch_size = sequence.size(0)
        feature_dim = sequence.size(-1)
        empty_feat = sequence.new_empty(batch_size, 0, feature_dim)
        empty_long = torch.empty(batch_size, 0, device=sequence.device, dtype=torch.long)
        return empty_feat, empty_feat, empty_long, empty_long

    return (
        torch.cat(src_chunks, dim=1),
        torch.cat(goal_chunks, dim=1),
        torch.cat(budget_chunks, dim=1),
        torch.cat(delta_chunks, dim=1),
    )


def _collect_cross_rollout_pairs(src_sequence, goal_sequence, expand_fn):
    src_chunks = []
    goal_chunks = []
    budget_chunks = []
    delta_chunks = []

    horizons = min(src_sequence.size(1), goal_sequence.size(1)) - 1
    for src_idx in range(horizons):
        src_i, goal_i, budget_i, delta_i = expand_fn(
            src_sequence[:, src_idx],
            goal_sequence[:, src_idx + 1 : horizons + 1],
        )
        if budget_i.numel() == 0:
            continue
        src_chunks.append(src_i)
        goal_chunks.append(goal_i)
        budget_chunks.append(budget_i)
        delta_chunks.append(delta_i)

    if not src_chunks:
        batch_size = src_sequence.size(0)
        feature_dim = src_sequence.size(-1)
        empty_feat = src_sequence.new_empty(batch_size, 0, feature_dim)
        empty_long = torch.empty(
            batch_size,
            0,
            device=src_sequence.device,
            dtype=torch.long,
        )
        return empty_feat, empty_feat, empty_long, empty_long

    return (
        torch.cat(src_chunks, dim=1),
        torch.cat(goal_chunks, dim=1),
        torch.cat(budget_chunks, dim=1),
        torch.cat(delta_chunks, dim=1),
    )


def _balanced_axis_weights(axis_ids, dtype):
    flat_ids = axis_ids.reshape(-1)
    counts = torch.bincount(flat_ids, minlength=int(flat_ids.max().item()) + 1)
    weights = counts[flat_ids].to(dtype).reciprocal()
    return weights.reshape_as(axis_ids)


def _resolve_pair_weighting(reach_cfg, prefix=""):
    mode_key = f"{prefix}horizon_weighting" if prefix else "horizon_weighting"
    power_key = f"{prefix}horizon_weight_power" if prefix else "horizon_weight_power"

    mode = reach_cfg.get(mode_key)
    if mode is None:
        mode = reach_cfg.get("horizon_weighting", "uniform")

    power = reach_cfg.get(power_key)
    if power is None:
        power = reach_cfg.get("horizon_weight_power", 1.0)

    return str(mode), float(power)


def _reachability_pair_weights(budget, delta, reach_cfg, dtype, prefix=""):
    mode, power = _resolve_pair_weighting(reach_cfg, prefix=prefix)

    if mode == "uniform":
        weights = torch.ones_like(delta, dtype=dtype)
    elif mode == "delta_balanced":
        weights = _balanced_axis_weights(delta, dtype)
    elif mode == "budget_balanced":
        weights = _balanced_axis_weights(budget, dtype)
    elif mode == "delta_linear":
        weights = delta.to(dtype)
    elif mode == "budget_linear":
        weights = budget.to(dtype)
    elif mode == "delta_power":
        weights = delta.to(dtype).pow(power)
    elif mode == "budget_power":
        weights = budget.to(dtype).pow(power)
    else:
        raise ValueError(f"Unknown reachability horizon weighting: {mode}")

    return weights / weights.mean().clamp_min(torch.finfo(dtype).eps)


def _weighted_bce(logits, targets, weights):
    if logits.numel() == 0:
        return logits.new_tensor(0.0)

    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    if weights is None:
        return losses.mean()

    scaled_weights = weights.to(losses.dtype)
    scaled_weights = scaled_weights / scaled_weights.mean().clamp_min(
        torch.finfo(losses.dtype).eps
    )
    return (losses * scaled_weights).mean()


def _adjacent_margin_ranking_loss(logits, margin):
    if logits.ndim != 2 or logits.size(1) < 2:
        return logits.new_tensor(0.0)
    deltas = logits[:, 1:] - logits[:, :-1]
    return torch.relu(logits.new_tensor(margin) - deltas).mean()


def _terminal_progress_pairs(src_seq, goal):
    if src_seq.size(1) == 0:
        batch_size = src_seq.size(0)
        feature_dim = src_seq.size(-1)
        empty_feat = src_seq.new_empty(batch_size, 0, feature_dim)
        empty_long = torch.empty(batch_size, 0, device=src_seq.device, dtype=torch.long)
        return empty_feat, empty_feat, empty_long, empty_long

    steps = src_seq.size(1)
    budgets = torch.arange(
        steps,
        0,
        -1,
        device=src_seq.device,
        dtype=torch.long,
    )
    budgets = budgets.unsqueeze(0).expand(src_seq.size(0), -1)
    delta = budgets.clone()
    goal_seq = goal.unsqueeze(1).expand(-1, steps, -1)
    return src_seq, goal_seq, budgets, delta


def _terminal_progress_negative_pairs(src_seq, goal):
    batch_size = src_seq.size(0)
    feature_dim = src_seq.size(-1)
    steps = src_seq.size(1)
    src_chunks = []
    goal_chunks = []
    budget_chunks = []
    delta_chunks = []

    for idx, required in enumerate(range(steps, 0, -1)):
        if required <= 1:
            continue
        budgets = torch.arange(1, required, device=src_seq.device, dtype=torch.long)
        num_budget = budgets.numel()
        src_chunks.append(src_seq[:, idx : idx + 1].expand(-1, num_budget, -1))
        goal_chunks.append(goal.unsqueeze(1).expand(-1, num_budget, -1))
        budget_chunks.append(budgets.unsqueeze(0).expand(batch_size, -1))
        delta_chunks.append(
            torch.full(
                (batch_size, num_budget),
                required,
                device=src_seq.device,
                dtype=torch.long,
            )
        )

    if not src_chunks:
        empty_feat = src_seq.new_empty(batch_size, 0, feature_dim)
        empty_long = torch.empty(batch_size, 0, device=src_seq.device, dtype=torch.long)
        return empty_feat, empty_feat, empty_long, empty_long

    return (
        torch.cat(src_chunks, dim=1),
        torch.cat(goal_chunks, dim=1),
        torch.cat(budget_chunks, dim=1),
        torch.cat(delta_chunks, dim=1),
    )


def _terminal_progress_objective(
    module,
    src_seq,
    goal,
    reach_cfg,
    hard_neg_weight,
    stats_prefix,
    rank_weight=0.0,
    rank_margin=0.0,
):
    zero = src_seq.new_tensor(0.0)
    src_pos, goal_pos, budget_pos, delta_pos = _terminal_progress_pairs(src_seq, goal)
    if budget_pos.numel() == 0:
        return zero, {}

    pos_logits = module.model.score_reachability(
        src_pos.reshape(-1, src_pos.size(-1)),
        goal_pos.reshape(-1, goal_pos.size(-1)),
        budget_pos.reshape(-1),
    )
    pos_weights = _reachability_pair_weights(
        budget_pos.reshape(-1),
        delta_pos.reshape(-1),
        reach_cfg,
        pos_logits.dtype,
        prefix="terminal_",
    )
    pos_loss = _weighted_bce(pos_logits, torch.ones_like(pos_logits), pos_weights)
    pos_logits_matrix = pos_logits.view(src_pos.size(0), src_pos.size(1))
    rank_loss = zero
    if rank_weight > 0.0:
        rank_loss = _adjacent_margin_ranking_loss(pos_logits_matrix, rank_margin)

    if hard_neg_weight > 0.0:
        src_hard, goal_hard, budget_hard, delta_hard = _terminal_progress_negative_pairs(
            src_seq,
            goal,
        )
        if budget_hard.numel() > 0:
            hard_logits = module.model.score_reachability(
                src_hard.reshape(-1, src_hard.size(-1)),
                goal_hard.reshape(-1, goal_hard.size(-1)),
                budget_hard.reshape(-1),
            )
            hard_weights = _reachability_pair_weights(
                budget_hard.reshape(-1),
                delta_hard.reshape(-1),
                reach_cfg,
                hard_logits.dtype,
                prefix="terminal_",
            )
            hard_neg_loss = _weighted_bce(
                hard_logits,
                torch.zeros_like(hard_logits),
                hard_weights,
            )
        else:
            hard_logits = pos_logits.new_zeros(0)
            hard_neg_loss = zero
    else:
        hard_logits = pos_logits.new_zeros(0)
        hard_neg_loss = zero

    if src_seq.size(0) > 1:
        perm = torch.randperm(src_seq.size(0), device=src_seq.device)
        if torch.equal(perm, torch.arange(src_seq.size(0), device=src_seq.device)):
            perm = torch.roll(perm, 1)
        goal_neg = goal_pos[perm]
        neg_logits = module.model.score_reachability(
            src_pos.reshape(-1, src_pos.size(-1)),
            goal_neg.reshape(-1, goal_neg.size(-1)),
            budget_pos.reshape(-1),
        )
        neg_loss = _weighted_bce(
            neg_logits,
            torch.zeros_like(neg_logits),
            pos_weights,
        )
    else:
        neg_logits = pos_logits.new_zeros(pos_logits.shape)
        neg_loss = zero

    neg_terms = neg_loss
    neg_weight = 1.0
    if hard_neg_weight > 0.0:
        neg_terms = neg_terms + hard_neg_weight * hard_neg_loss
        neg_weight = neg_weight + hard_neg_weight

    loss = 0.5 * (pos_loss + neg_terms / neg_weight)
    if rank_weight > 0.0:
        loss = loss + rank_weight * rank_loss
    stats = {
        f"{stats_prefix}_pos_prob": torch.sigmoid(pos_logits).mean().detach(),
        f"{stats_prefix}_neg_prob": torch.sigmoid(neg_logits).mean().detach(),
        f"{stats_prefix}_pos_delta": (
            pos_logits_matrix[:, 1:] - pos_logits_matrix[:, :-1]
        ).mean().detach()
        if pos_logits_matrix.size(1) > 1
        else zero.detach(),
    }
    if rank_weight > 0.0:
        stats[f"{stats_prefix}_rank_loss"] = rank_loss.detach()
    if hard_logits.numel() > 0:
        stats[f"{stats_prefix}_hard_neg_prob"] = torch.sigmoid(hard_logits).mean().detach()
    return loss, stats


def _terminal_distance_pairs(src_seq, goal):
    if src_seq.size(1) == 0:
        batch_size = src_seq.size(0)
        feature_dim = src_seq.size(-1)
        empty_feat = src_seq.new_empty(batch_size, 0, feature_dim)
        empty_long = torch.empty(batch_size, 0, device=src_seq.device, dtype=torch.long)
        return empty_feat, empty_feat, empty_long

    steps = src_seq.size(1)
    delta = torch.arange(
        steps,
        0,
        -1,
        device=src_seq.device,
        dtype=torch.long,
    )
    delta = delta.unsqueeze(0).expand(src_seq.size(0), -1)
    goal_seq = goal.unsqueeze(1).expand(-1, steps, -1)
    return src_seq, goal_seq, delta


def _terminal_distance_objective(
    module,
    src_seq,
    goal,
    neg_weight,
    stats_prefix,
):
    zero = src_seq.new_tensor(0.0)
    src_pos, goal_pos, delta_pos = _terminal_distance_pairs(src_seq, goal)
    if delta_pos.numel() == 0:
        return zero, {}

    pos_logits = module.model.score_temporal_distance(
        src_pos.reshape(-1, src_pos.size(-1)),
        goal_pos.reshape(-1, goal_pos.size(-1)),
    )
    pos_loss = torch.nn.functional.cross_entropy(pos_logits, delta_pos.reshape(-1))

    if src_seq.size(0) > 1 and neg_weight > 0.0:
        perm = torch.randperm(src_seq.size(0), device=src_seq.device)
        if torch.equal(perm, torch.arange(src_seq.size(0), device=src_seq.device)):
            perm = torch.roll(perm, 1)
        neg_logits = module.model.score_temporal_distance(
            src_pos.reshape(-1, src_pos.size(-1)),
            goal_pos[perm].reshape(-1, goal_pos.size(-1)),
        )
        neg_labels = torch.zeros(
            neg_logits.size(0),
            device=src_seq.device,
            dtype=torch.long,
        )
        neg_loss = torch.nn.functional.cross_entropy(neg_logits, neg_labels)
        loss = (pos_loss + neg_weight * neg_loss) / (1.0 + neg_weight)
    else:
        neg_logits = pos_logits.new_zeros((0, pos_logits.size(-1)))
        loss = pos_loss

    class_values = torch.arange(pos_logits.size(-1), device=src_seq.device, dtype=src_seq.dtype)
    pos_expected = (
        torch.softmax(pos_logits, dim=-1) * class_values.unsqueeze(0)
    ).sum(dim=-1)
    stats = {
        f"{stats_prefix}_expected": pos_expected.mean().detach(),
    }
    if neg_logits.numel() > 0:
        stats[f"{stats_prefix}_neg_zero_prob"] = (
            torch.softmax(neg_logits, dim=-1)[:, 0].mean().detach()
        )
    return loss, stats


def reachability_loss(module, emb, pred_emb, cfg):
    reach_cfg = cfg.loss.get("reachability")
    zero = emb.new_tensor(0.0)
    if reach_cfg is None or not reach_cfg.enabled:
        return zero, {}
    if getattr(module.model, "reachability_head", None) is None:
        return zero, {}

    ctx_idx = cfg.wm.history_size - 1
    available_horizon = emb.size(1) - (ctx_idx + 1)
    max_horizon = min(int(reach_cfg.max_horizon), int(available_horizon))
    if max_horizon < 1:
        return zero, {}

    src = emb[:, ctx_idx]
    source_mode = str(reach_cfg.get("source_mode", "context_only"))
    if source_mode == "context_only":
        future_true = emb[:, ctx_idx + 1 : ctx_idx + 1 + max_horizon]
        src_true, goal_true, budget_true, delta_true = _expand_reach_pairs(src, future_true)
    elif source_mode == "dense_rollout":
        true_sequence = emb[:, ctx_idx : ctx_idx + 1 + max_horizon]
        src_true, goal_true, budget_true, delta_true = _collect_rollout_pairs(
            true_sequence,
            _expand_reach_pairs,
        )
    else:
        raise ValueError(f"Unknown reachability source mode: {source_mode}")

    pos_logits = module.model.score_reachability(
        src_true.reshape(-1, src_true.size(-1)),
        goal_true.reshape(-1, goal_true.size(-1)),
        budget_true.reshape(-1),
    )
    pos_weights = _reachability_pair_weights(
        budget_true.reshape(-1),
        delta_true.reshape(-1),
        reach_cfg,
        pos_logits.dtype,
    )
    pos_loss = _weighted_bce(pos_logits, torch.ones_like(pos_logits), pos_weights)

    temporal_neg_weight = float(reach_cfg.get("temporal_neg_weight", 1.0))
    if temporal_neg_weight > 0.0:
        if source_mode == "context_only":
            src_hard, goal_hard, budget_hard, delta_hard = _expand_temporal_negative_pairs(
                src,
                future_true,
            )
        else:
            src_hard, goal_hard, budget_hard, delta_hard = _collect_rollout_pairs(
                true_sequence,
                _expand_temporal_negative_pairs,
            )
        if budget_hard.numel() > 0:
            hard_logits = module.model.score_reachability(
                src_hard.reshape(-1, src_hard.size(-1)),
                goal_hard.reshape(-1, goal_hard.size(-1)),
                budget_hard.reshape(-1),
            )
            hard_weights = _reachability_pair_weights(
                budget_hard.reshape(-1),
                delta_hard.reshape(-1),
                reach_cfg,
                hard_logits.dtype,
            )
            hard_neg_loss = _weighted_bce(
                hard_logits,
                torch.zeros_like(hard_logits),
                hard_weights,
            )
        else:
            hard_logits = pos_logits.new_zeros(0)
            hard_neg_loss = zero
    else:
        hard_logits = pos_logits.new_zeros(0)
        hard_neg_loss = zero

    if src.size(0) > 1:
        perm = torch.randperm(src.size(0), device=src.device)
        if torch.equal(perm, torch.arange(src.size(0), device=src.device)):
            perm = torch.roll(perm, 1)
        goal_true_neg = goal_true[perm]
        neg_logits = module.model.score_reachability(
            src_true.reshape(-1, src_true.size(-1)),
            goal_true_neg.reshape(-1, goal_true_neg.size(-1)),
            budget_true.reshape(-1),
        )
        neg_loss = _weighted_bce(
            neg_logits,
            torch.zeros_like(neg_logits),
            pos_weights,
        )
    else:
        neg_logits = pos_logits.new_zeros(pos_logits.shape)
        neg_loss = zero

    neg_terms = neg_loss
    neg_weight = 1.0
    if temporal_neg_weight > 0.0:
        neg_terms = neg_terms + temporal_neg_weight * hard_neg_loss
        neg_weight = neg_weight + temporal_neg_weight

    loss = 0.5 * (pos_loss + neg_terms / neg_weight)
    stats = {
        "reach_pos_prob": torch.sigmoid(pos_logits).mean().detach(),
        "reach_neg_prob": torch.sigmoid(neg_logits).mean().detach(),
    }
    if hard_logits.numel() > 0:
        stats["reach_temporal_neg_prob"] = torch.sigmoid(hard_logits).mean().detach()

    pred_weight = float(reach_cfg.pred_weight)
    pred_available = min(pred_emb.size(1), max_horizon)
    if pred_weight > 0.0 and pred_available > 0:
        if source_mode == "context_only":
            future_pred = pred_emb[:, :pred_available]
            src_pred, goal_pred, budget_pred, delta_pred = _expand_reach_pairs(
                src,
                future_pred,
            )
        else:
            pred_sequence = torch.cat([src.unsqueeze(1), pred_emb[:, :pred_available]], dim=1)
            src_pred, goal_pred, budget_pred, delta_pred = _collect_rollout_pairs(
                pred_sequence,
                _expand_reach_pairs,
            )
        pred_pos_logits = module.model.score_reachability(
            src_pred.reshape(-1, src_pred.size(-1)),
            goal_pred.reshape(-1, goal_pred.size(-1)),
            budget_pred.reshape(-1),
        )
        pred_pos_weights = _reachability_pair_weights(
            budget_pred.reshape(-1),
            delta_pred.reshape(-1),
            reach_cfg,
            pred_pos_logits.dtype,
        )
        pred_pos_loss = _weighted_bce(
            pred_pos_logits,
            torch.ones_like(pred_pos_logits),
            pred_pos_weights,
        )

        pred_temporal_neg_weight = float(
            reach_cfg.get("pred_temporal_neg_weight", temporal_neg_weight)
        )
        if pred_temporal_neg_weight > 0.0:
            if source_mode == "context_only":
                src_pred_hard, goal_pred_hard, budget_pred_hard, delta_pred_hard = (
                    _expand_temporal_negative_pairs(
                        src,
                        future_pred,
                    )
                )
            else:
                src_pred_hard, goal_pred_hard, budget_pred_hard, delta_pred_hard = (
                    _collect_rollout_pairs(
                        pred_sequence,
                        _expand_temporal_negative_pairs,
                    )
                )
            if budget_pred_hard.numel() > 0:
                pred_hard_logits = module.model.score_reachability(
                    src_pred_hard.reshape(-1, src_pred_hard.size(-1)),
                    goal_pred_hard.reshape(-1, goal_pred_hard.size(-1)),
                    budget_pred_hard.reshape(-1),
                )
                pred_hard_weights = _reachability_pair_weights(
                    budget_pred_hard.reshape(-1),
                    delta_pred_hard.reshape(-1),
                    reach_cfg,
                    pred_hard_logits.dtype,
                )
                pred_hard_neg_loss = _weighted_bce(
                    pred_hard_logits,
                    torch.zeros_like(pred_hard_logits),
                    pred_hard_weights,
                )
            else:
                pred_hard_logits = pred_pos_logits.new_zeros(0)
                pred_hard_neg_loss = zero
        else:
            pred_hard_logits = pred_pos_logits.new_zeros(0)
            pred_hard_neg_loss = zero

        if src.size(0) > 1:
            perm = torch.randperm(src.size(0), device=src.device)
            if torch.equal(perm, torch.arange(src.size(0), device=src.device)):
                perm = torch.roll(perm, 1)
            goal_pred_neg = goal_pred[perm]
            pred_neg_logits = module.model.score_reachability(
                src_pred.reshape(-1, src_pred.size(-1)),
                goal_pred_neg.reshape(-1, goal_pred_neg.size(-1)),
                budget_pred.reshape(-1),
            )
            pred_neg_loss = _weighted_bce(
                pred_neg_logits,
                torch.zeros_like(pred_neg_logits),
                pred_pos_weights,
            )
        else:
            pred_neg_logits = pred_pos_logits.new_zeros(pred_pos_logits.shape)
            pred_neg_loss = zero

        pred_neg_terms = pred_neg_loss
        pred_neg_weight = 1.0
        if pred_temporal_neg_weight > 0.0:
            pred_neg_terms = pred_neg_terms + pred_temporal_neg_weight * pred_hard_neg_loss
            pred_neg_weight = pred_neg_weight + pred_temporal_neg_weight

        pred_reach_loss = 0.5 * (pred_pos_loss + pred_neg_terms / pred_neg_weight)
        loss = loss + pred_weight * pred_reach_loss
        stats["reach_pred_pos_prob"] = torch.sigmoid(pred_pos_logits).mean().detach()
        stats["reach_pred_neg_prob"] = torch.sigmoid(pred_neg_logits).mean().detach()
        if pred_hard_logits.numel() > 0:
            stats["reach_pred_temporal_neg_prob"] = torch.sigmoid(pred_hard_logits).mean().detach()

    rollout_teacher_weight = float(
        reach_cfg.get("rollout_teacher_weight", 0.0)
    )
    rollout_source_teacher_weight = float(
        reach_cfg.get("rollout_source_teacher_weight", 0.0)
    )
    if rollout_teacher_weight > 0.0 and pred_available > 0:
        true_rollout_sequence = torch.cat(
            [
                src.unsqueeze(1),
                emb[:, ctx_idx + 1 : ctx_idx + 1 + pred_available],
            ],
            dim=1,
        )
        pred_rollout_sequence = torch.cat(
            [src.unsqueeze(1), pred_emb[:, :pred_available]],
            dim=1,
        )
        teacher_src_true, teacher_goal_true, teacher_budget, teacher_delta = (
            _collect_rollout_pairs(
                true_rollout_sequence,
                _expand_reach_pairs,
            )
        )
        teacher_src_pred, teacher_goal_pred, _, _ = _collect_rollout_pairs(
            pred_rollout_sequence,
            _expand_reach_pairs,
        )
        if teacher_budget.numel() > 0:
            teacher_true_logits = module.model.score_reachability(
                teacher_src_true.reshape(-1, teacher_src_true.size(-1)),
                teacher_goal_true.reshape(-1, teacher_goal_true.size(-1)),
                teacher_budget.reshape(-1),
            )
            teacher_pred_logits = module.model.score_reachability(
                teacher_src_pred.reshape(-1, teacher_src_pred.size(-1)),
                teacher_goal_pred.reshape(-1, teacher_goal_pred.size(-1)),
                teacher_budget.reshape(-1),
            )
            teacher_pair_weights = _reachability_pair_weights(
                teacher_budget.reshape(-1),
                teacher_delta.reshape(-1),
                reach_cfg,
                teacher_pred_logits.dtype,
            )
            teacher_targets = torch.sigmoid(teacher_true_logits.detach())
            teacher_loss = _weighted_bce(
                teacher_pred_logits,
                teacher_targets,
                teacher_pair_weights,
            )
            loss = loss + rollout_teacher_weight * teacher_loss
            stats["reach_rollout_teacher_loss"] = teacher_loss.detach()
            stats["reach_rollout_teacher_gap"] = (
                torch.sigmoid(teacher_pred_logits) - teacher_targets
            ).abs().mean().detach()

    if rollout_source_teacher_weight > 0.0 and pred_available > 0:
        true_rollout_sequence = torch.cat(
            [
                src.unsqueeze(1),
                emb[:, ctx_idx + 1 : ctx_idx + 1 + pred_available],
            ],
            dim=1,
        )
        pred_rollout_sequence = torch.cat(
            [src.unsqueeze(1), pred_emb[:, :pred_available]],
            dim=1,
        )
        teacher_src_true, teacher_goal_true, teacher_budget, teacher_delta = (
            _collect_cross_rollout_pairs(
                true_rollout_sequence,
                true_rollout_sequence,
                _expand_reach_pairs,
            )
        )
        teacher_src_pred, teacher_goal_shared, _, _ = _collect_cross_rollout_pairs(
            pred_rollout_sequence,
            true_rollout_sequence,
            _expand_reach_pairs,
        )
        if teacher_budget.numel() > 0:
            teacher_true_logits = module.model.score_reachability(
                teacher_src_true.reshape(-1, teacher_src_true.size(-1)),
                teacher_goal_true.reshape(-1, teacher_goal_true.size(-1)),
                teacher_budget.reshape(-1),
            )
            teacher_pred_logits = module.model.score_reachability(
                teacher_src_pred.reshape(-1, teacher_src_pred.size(-1)),
                teacher_goal_shared.reshape(-1, teacher_goal_shared.size(-1)),
                teacher_budget.reshape(-1),
            )
            teacher_pair_weights = _reachability_pair_weights(
                teacher_budget.reshape(-1),
                teacher_delta.reshape(-1),
                reach_cfg,
                teacher_pred_logits.dtype,
            )
            teacher_targets = torch.sigmoid(teacher_true_logits.detach())
            teacher_loss = _weighted_bce(
                teacher_pred_logits,
                teacher_targets,
                teacher_pair_weights,
            )
            loss = loss + rollout_source_teacher_weight * teacher_loss
            stats["reach_rollout_source_teacher_loss"] = teacher_loss.detach()
            stats["reach_rollout_source_teacher_gap"] = (
                torch.sigmoid(teacher_pred_logits) - teacher_targets
            ).abs().mean().detach()

    terminal_weight = float(reach_cfg.get("terminal_weight", 0.0))
    terminal_pred_weight = float(reach_cfg.get("terminal_pred_weight", 0.0))
    terminal_neg_weight = float(
        reach_cfg.get("terminal_neg_weight", temporal_neg_weight)
    )
    terminal_rank_weight = float(reach_cfg.get("terminal_rank_weight", 0.0))
    terminal_pred_rank_weight = float(
        reach_cfg.get("terminal_pred_rank_weight", terminal_rank_weight)
    )
    terminal_rank_margin = float(reach_cfg.get("terminal_rank_margin", 0.0))
    terminal_pred_align_weight = float(
        reach_cfg.get("terminal_pred_align_weight", 0.0)
    )
    terminal_pred_over_weight = float(
        reach_cfg.get("terminal_pred_over_weight", 0.0)
    )
    terminal_pred_over_margin = float(
        reach_cfg.get("terminal_pred_over_margin", 0.0)
    )
    terminal_pred_gap_weight = float(
        reach_cfg.get("terminal_pred_gap_weight", 0.0)
    )
    current_horizon = min(max_horizon, pred_emb.size(1), available_horizon)
    if current_horizon > 0 and (terminal_weight > 0.0 or terminal_pred_weight > 0.0):
        terminal_goal = emb[:, ctx_idx + current_horizon]
        true_sources = torch.cat(
            [src.unsqueeze(1), emb[:, ctx_idx + 1 : ctx_idx + current_horizon]],
            dim=1,
        )
        if terminal_weight > 0.0:
            term_true_loss, term_true_stats = _terminal_progress_objective(
                module,
                true_sources,
                terminal_goal,
                reach_cfg,
                terminal_neg_weight,
                "reach_terminal",
                rank_weight=terminal_rank_weight,
                rank_margin=terminal_rank_margin,
            )
            loss = loss + terminal_weight * term_true_loss
            stats.update(term_true_stats)

        if current_horizon > 1:
            pred_sources = pred_emb[:, : current_horizon - 1]
            if terminal_pred_weight > 0.0:
                term_pred_loss, term_pred_stats = _terminal_progress_objective(
                    module,
                    pred_sources,
                    terminal_goal,
                    reach_cfg,
                    terminal_neg_weight,
                    "reach_terminal_pred",
                    rank_weight=terminal_pred_rank_weight,
                    rank_margin=terminal_rank_margin,
                )
                loss = loss + terminal_pred_weight * term_pred_loss
                stats.update(term_pred_stats)

            if (
                terminal_pred_align_weight > 0.0
                or terminal_pred_over_weight > 0.0
                or terminal_pred_gap_weight > 0.0
            ):
                align_budgets = torch.arange(
                    current_horizon - 1,
                    0,
                    -1,
                    device=emb.device,
                    dtype=torch.long,
                )
                align_budgets = align_budgets.unsqueeze(0).expand(emb.size(0), -1)
                align_goal = terminal_goal.unsqueeze(1).expand(-1, current_horizon - 1, -1)
                true_align_logits = module.model.score_reachability(
                    true_sources[:, 1:].reshape(-1, true_sources.size(-1)),
                    align_goal.reshape(-1, align_goal.size(-1)),
                    align_budgets.reshape(-1),
                )
                pred_align_logits = module.model.score_reachability(
                    pred_sources.reshape(-1, pred_sources.size(-1)),
                    align_goal.reshape(-1, align_goal.size(-1)),
                    align_budgets.reshape(-1),
                )
                align_weights = _reachability_pair_weights(
                    align_budgets.reshape(-1),
                    align_budgets.reshape(-1),
                    reach_cfg,
                    pred_align_logits.dtype,
                    prefix="terminal_",
                )
                align_targets = torch.sigmoid(true_align_logits.detach())
                pred_align_probs = torch.sigmoid(pred_align_logits)
                align_targets_matrix = align_targets.view(emb.size(0), current_horizon - 1)
                pred_align_matrix = pred_align_probs.view(emb.size(0), current_horizon - 1)
                align_weights_matrix = align_weights.view(emb.size(0), current_horizon - 1)
                if terminal_pred_align_weight > 0.0:
                    term_align_loss = _weighted_bce(
                        pred_align_logits,
                        align_targets,
                        align_weights,
                    )
                    loss = loss + terminal_pred_align_weight * term_align_loss
                    stats["reach_terminal_pred_align_loss"] = term_align_loss.detach()
                    stats["reach_terminal_pred_align_gap"] = (
                        pred_align_probs - align_targets
                    ).abs().mean().detach()
                if terminal_pred_over_weight > 0.0:
                    over_margin = pred_align_probs.new_tensor(terminal_pred_over_margin)
                    over_errors = torch.relu(pred_align_probs - align_targets - over_margin)
                    scaled_weights = align_weights.to(over_errors.dtype)
                    scaled_weights = scaled_weights / scaled_weights.mean().clamp_min(
                        torch.finfo(over_errors.dtype).eps
                    )
                    term_over_loss = (over_errors * scaled_weights).mean()
                    loss = loss + terminal_pred_over_weight * term_over_loss
                    stats["reach_terminal_pred_over_loss"] = term_over_loss.detach()
                    stats["reach_terminal_pred_over_rate"] = (
                        pred_align_probs > (align_targets + terminal_pred_over_margin)
                    ).float().mean().detach()
                if terminal_pred_gap_weight > 0.0 and current_horizon > 2:
                    true_gaps = align_targets_matrix[:, 1:] - align_targets_matrix[:, :-1]
                    pred_gaps = pred_align_matrix[:, 1:] - pred_align_matrix[:, :-1]
                    gap_errors = (pred_gaps - true_gaps).pow(2)
                    gap_weights = 0.5 * (
                        align_weights_matrix[:, 1:] + align_weights_matrix[:, :-1]
                    )
                    scaled_gap_weights = gap_weights / gap_weights.mean().clamp_min(
                        torch.finfo(gap_errors.dtype).eps
                    )
                    term_gap_loss = (gap_errors * scaled_gap_weights).mean()
                    loss = loss + terminal_pred_gap_weight * term_gap_loss
                    stats["reach_terminal_pred_gap_loss"] = term_gap_loss.detach()
                    stats["reach_terminal_pred_gap_absdiff"] = (
                        pred_gaps - true_gaps
                    ).abs().mean().detach()

    return loss, stats


def grounding_loss(module, batch, emb, pred_emb, cfg):
    ground_cfg = cfg.loss.get("grounding")
    zero = emb.new_tensor(0.0)
    head = getattr(module.model, "grounding_head", None)
    if ground_cfg is None or not ground_cfg.enabled or head is None:
        return zero, {}

    target_key = ground_cfg.target_key
    if target_key not in batch:
        return zero, {}

    target = batch[target_key].float().to(emb.device)
    loss = zero
    stats = {}

    obs_weight = float(ground_cfg.obs_weight)
    if obs_weight > 0.0:
        obs_pred = module.model.decode_grounding(emb)
        obs_loss = (obs_pred - target).pow(2).mean()
        loss = loss + obs_weight * obs_loss
        stats["ground_obs_loss"] = obs_loss.detach()

    pred_weight = float(ground_cfg.pred_weight)
    ctx_len = cfg.wm.history_size
    pred_horizon = min(pred_emb.size(1), target.size(1) - ctx_len)
    if pred_weight > 0.0 and pred_horizon > 0:
        pred_target = target[:, ctx_len : ctx_len + pred_horizon].to(pred_emb.device)
        pred_decoded = module.model.decode_grounding(pred_emb[:, :pred_horizon])
        pred_loss = (pred_decoded - pred_target).pow(2).mean()
        loss = loss + pred_weight * pred_loss
        stats["ground_pred_loss"] = pred_loss.detach()

    return loss, stats


def temporal_distance_loss(module, emb, pred_emb, cfg):
    td_cfg = cfg.loss.get("temporal_distance")
    zero = emb.new_tensor(0.0)
    head = getattr(module.model, "temporal_distance_head", None)
    if td_cfg is None or not td_cfg.enabled or head is None:
        return zero, {}

    ctx_idx = cfg.wm.history_size - 1
    available_horizon = emb.size(1) - (ctx_idx + 1)
    max_horizon = min(int(td_cfg.max_horizon), int(available_horizon))
    if max_horizon < 1:
        return zero, {}

    src = emb[:, ctx_idx]
    future_true = emb[:, ctx_idx + 1 : ctx_idx + 1 + max_horizon]
    src_true, goal_true, delta_true = _expand_temporal_distance_pairs(src, future_true)
    pos_logits = module.model.score_temporal_distance(src_true, goal_true)
    pos_loss = torch.nn.functional.cross_entropy(pos_logits, delta_true)

    neg_weight = float(td_cfg.neg_weight)
    if src.size(0) > 1:
        perm = torch.randperm(src.size(0), device=src.device)
        if torch.equal(perm, torch.arange(src.size(0), device=src.device)):
            perm = torch.roll(perm, 1)
        src_neg = src.unsqueeze(1).expand(-1, max_horizon, -1).reshape(-1, src.size(-1))
        goal_neg = future_true[perm].reshape(-1, future_true.size(-1))
        neg_logits = module.model.score_temporal_distance(src_neg, goal_neg)
        neg_labels = torch.zeros(src_neg.size(0), device=src.device, dtype=torch.long)
        neg_loss = torch.nn.functional.cross_entropy(neg_logits, neg_labels)
    else:
        neg_logits = pos_logits.new_zeros((0, max_horizon + 1))
        neg_loss = zero

    pair_weight = float(td_cfg.get("pair_weight", 1.0))
    loss = zero
    if pair_weight > 0.0:
        pair_loss = pos_loss
        denom = 1.0
        if neg_weight > 0.0 and src.size(0) > 1:
            pair_loss = pair_loss + neg_weight * neg_loss
            denom += neg_weight
        pair_loss = pair_loss / denom
        loss = loss + pair_weight * pair_loss

    class_values = torch.arange(pos_logits.size(-1), device=src.device, dtype=emb.dtype)
    pos_expected = (
        torch.softmax(pos_logits, dim=-1) * class_values.unsqueeze(0)
    ).sum(dim=-1)
    stats = {
        "td_pos_expected": pos_expected.mean().detach(),
    }
    if neg_logits.numel() > 0:
        stats["td_neg_zero_prob"] = (
            torch.softmax(neg_logits, dim=-1)[:, 0].mean().detach()
        )

    pred_weight = float(td_cfg.pred_weight)
    pred_available = min(pred_emb.size(1), max_horizon)
    if pred_weight > 0.0 and pred_available > 0:
        future_pred = pred_emb[:, :pred_available]
        src_pred, goal_pred, delta_pred = _expand_temporal_distance_pairs(src, future_pred)
        pred_pos_logits = module.model.score_temporal_distance(src_pred, goal_pred)
        pred_pos_loss = torch.nn.functional.cross_entropy(pred_pos_logits, delta_pred)

        pred_loss = pred_pos_loss
        pred_denom = 1.0
        if neg_weight > 0.0 and src.size(0) > 1:
            src_pred_neg = src.unsqueeze(1).expand(-1, pred_available, -1).reshape(
                -1, src.size(-1)
            )
            goal_pred_neg = future_pred[perm].reshape(-1, future_pred.size(-1))
            pred_neg_logits = module.model.score_temporal_distance(
                src_pred_neg,
                goal_pred_neg,
            )
            pred_neg_labels = torch.zeros(
                src_pred_neg.size(0), device=src.device, dtype=torch.long
            )
            pred_neg_loss = torch.nn.functional.cross_entropy(
                pred_neg_logits,
                pred_neg_labels,
            )
            pred_loss = pred_loss + neg_weight * pred_neg_loss
            pred_denom += neg_weight
            stats["td_pred_neg_zero_prob"] = (
                torch.softmax(pred_neg_logits, dim=-1)[:, 0].mean().detach()
            )

        pred_loss = pred_loss / pred_denom
        loss = loss + pred_weight * pred_loss

        pred_class_values = torch.arange(
            pred_pos_logits.size(-1),
            device=src.device,
            dtype=emb.dtype,
        )
        pred_expected = (
            torch.softmax(pred_pos_logits, dim=-1) * pred_class_values.unsqueeze(0)
        ).sum(dim=-1)
        stats["td_pred_pos_expected"] = pred_expected.mean().detach()

    terminal_weight = float(td_cfg.get("terminal_weight", 0.0))
    terminal_pred_weight = float(td_cfg.get("terminal_pred_weight", 0.0))
    terminal_neg_weight = float(td_cfg.get("terminal_neg_weight", neg_weight))
    current_horizon = min(max_horizon, pred_emb.size(1), available_horizon)
    if current_horizon > 0 and (terminal_weight > 0.0 or terminal_pred_weight > 0.0):
        terminal_goal = emb[:, ctx_idx + current_horizon]
        true_sources = torch.cat(
            [src.unsqueeze(1), emb[:, ctx_idx + 1 : ctx_idx + current_horizon]],
            dim=1,
        )
        if terminal_weight > 0.0:
            term_true_loss, term_true_stats = _terminal_distance_objective(
                module,
                true_sources,
                terminal_goal,
                terminal_neg_weight,
                "td_terminal",
            )
            loss = loss + terminal_weight * term_true_loss
            stats.update(term_true_stats)

        if terminal_pred_weight > 0.0 and current_horizon > 1:
            pred_sources = pred_emb[:, : current_horizon - 1]
            term_pred_loss, term_pred_stats = _terminal_distance_objective(
                module,
                pred_sources,
                terminal_goal,
                terminal_neg_weight,
                "td_terminal_pred",
            )
            loss = loss + terminal_pred_weight * term_pred_loss
            stats.update(term_pred_stats)

    return loss, stats


def maybe_freeze_for_reachability(model, cfg):
    reach_cfg = cfg.loss.get("reachability")
    if reach_cfg is None or not reach_cfg.enabled:
        return
    if not reach_cfg.get("freeze_base", False):
        return

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("reachability_head.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(
        "Frozen base world model for reachability-only training "
        f"(trainable={trainable}, frozen={frozen})"
    )


def rcaux_forward(self, batch, stage, cfg):
    """Encode observations, predict future states, compute RC-aux losses."""

    ctx_len = cfg.wm.history_size
    reg_type = cfg.loss.get("reg_type", "sigreg")
    reg_cfg = cfg.loss.get(reg_type)
    lambd = reg_cfg.weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)

    emb = output["emb"]
    act_emb = output["act_emb"]
    pred_type = cfg.loss.get("prediction_type", "one_step")
    reach_cfg = cfg.loss.get("reachability")

    if pred_type == "one_step":
        pred_loss, pred_emb, _, step_losses = one_step_prediction_loss(
            self.model, emb, act_emb, ctx_len
        )
        rollout_horizon = 1
        anchor_loss = emb.new_tensor(0.0)
        prefix_loss = emb.new_tensor(0.0)
    elif pred_type == "multi_horizon":
        pred_loss, pred_emb, _, step_losses, rollout_horizon = multi_horizon_prediction_loss(
            self, emb, act_emb, cfg, stage
        )

        anchor_weight = float(cfg.loss.rollout.get("one_step_anchor_weight", 0.0))
        if anchor_weight > 0.0:
            anchor_loss, _, _, _ = one_step_prediction_loss(self.model, emb, act_emb, ctx_len)
        else:
            anchor_loss = emb.new_tensor(0.0)

        prefix_weight = float(cfg.loss.rollout.get("prefix_weight", 0.0))
        prefix_horizon = int(cfg.loss.rollout.get("prefix_horizon", 0))
        available_horizon = emb.size(1) - ctx_len
        prefix_horizon = min(prefix_horizon, int(available_horizon))
        if prefix_weight > 0.0 and prefix_horizon > 0:
            prefix_loss, _, _, _ = open_loop_prediction_loss(
                self, emb, act_emb, cfg, prefix_horizon
            )
        else:
            prefix_loss = emb.new_tensor(0.0)
    else:
        raise ValueError(f"Unknown prediction_type: {pred_type}")

    output["pred_loss"] = pred_loss
    output["anchor_loss"] = anchor_loss
    output["prefix_loss"] = prefix_loss
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["reach_loss"], reach_stats = reachability_loss(self, emb, pred_emb, cfg)
    output["ground_loss"], ground_stats = grounding_loss(self, batch, emb, pred_emb, cfg)
    output["td_loss"], td_stats = temporal_distance_loss(self, emb, pred_emb, cfg)
    reach_weight = float(reach_cfg.weight) if reach_cfg is not None else 0.0
    ground_cfg = cfg.loss.get("grounding")
    ground_weight = float(ground_cfg.weight) if ground_cfg is not None else 0.0
    td_cfg = cfg.loss.get("temporal_distance")
    td_weight = float(td_cfg.weight) if td_cfg is not None else 0.0

    anchor_weight = float(cfg.loss.rollout.get("one_step_anchor_weight", 0.0))
    prefix_weight = float(cfg.loss.rollout.get("prefix_weight", 0.0))
    output["loss"] = (
        output["pred_loss"]
        + anchor_weight * output["anchor_loss"]
        + prefix_weight * output["prefix_loss"]
        + lambd * output["sigreg_loss"]
        + reach_weight * output["reach_loss"]
        + ground_weight * output["ground_loss"]
        + td_weight * output["td_loss"]
    )
    output["pred_last_loss"] = step_losses[-1]
    output["pred_horizon"] = emb.new_tensor(float(rollout_horizon))

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    losses_dict[f"{stage}/pred_horizon"] = output["pred_horizon"].detach()
    for key, value in reach_stats.items():
        losses_dict[f"{stage}/{key}"] = value.detach()
    for key, value in ground_stats.items():
        losses_dict[f"{stage}/{key}"] = value.detach()
    for key, value in td_stats.items():
        losses_dict[f"{stage}/{key}"] = value.detach()
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output
