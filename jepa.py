import os

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        reachability_head=None,
        grounding_head=None,
        temporal_distance_head=None,
        use_reachability_cost=False,
        reachability_cost_weight=0.0,
        use_temporal_distance_cost=False,
        temporal_distance_cost_weight=0.0,
        temporal_distance_cost_reduce="min",
        latent_cost_weight=1.0,
        goal_cost_reduce="terminal",
        goal_cost_softmin_temperature=1.0,
        action_l2_cost_weight=0.0,
        action_smooth_cost_weight=0.0,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.reachability_head = reachability_head
        self.grounding_head = grounding_head
        self.temporal_distance_head = temporal_distance_head
        self.use_reachability_cost = use_reachability_cost
        self.reachability_cost_weight = reachability_cost_weight
        self.use_temporal_distance_cost = use_temporal_distance_cost
        self.temporal_distance_cost_weight = temporal_distance_cost_weight
        self.temporal_distance_cost_reduce = temporal_distance_cost_reduce
        self.latent_cost_weight = latent_cost_weight
        self.goal_cost_reduce = goal_cost_reduce
        self.goal_cost_softmin_temperature = goal_cost_softmin_temperature
        self.action_l2_cost_weight = action_l2_cost_weight
        self.action_smooth_cost_weight = action_smooth_cost_weight

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info['pixels'].float()
        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0).unsqueeze(0)
        elif pixels.ndim == 4:
            pixels = pixels.unsqueeze(1)
        elif pixels.ndim != 5:
            raise ValueError(
                f"expected pixels with 3/4/5 dims, got shape {tuple(pixels.shape)}"
            )
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        pixels = info["pixels"]
        legacy_h_from_dim2 = os.getenv("LEWM_LEGACY_H_FROM_DIM2") == "1"
        if pixels.ndim == 5:
            H = pixels.size(2) if legacy_h_from_dim2 else 1
        elif pixels.ndim == 6:
            H = pixels.size(2)
        else:
            raise ValueError(
                f"expected pixels with 5/6 dims inside rollout, got shape {tuple(pixels.shape)}"
            )
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]
            act_trunc = act_emb[:, -HS:]
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)

            next_act = act_future[:, t : t + 1, :]
            act = torch.cat([act, next_act], dim=1)

        act_emb = self.action_encoder(act)
        emb_trunc = emb[:, -HS:]
        act_trunc = act_emb[:, -HS:]
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def rollout_open_loop(
        self,
        emb_history,
        act_history,
        future_act_emb,
        horizon: int,
        history_size: int | None = None,
    ):
        """Autoregressively predict future embeddings from latent/action embeddings."""

        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

        history_size = history_size or self.predictor.pos_embedding.size(1)
        cur_emb = emb_history
        cur_act = act_history
        preds = []

        for step in range(horizon):
            next_pred = self.predict(
                cur_emb[:, -history_size:],
                cur_act[:, -history_size:],
            )[:, -1:]
            preds.append(next_pred)

            if step + 1 < horizon:
                next_act = future_act_emb[:, step : step + 1]
                cur_emb = torch.cat([cur_emb, next_pred], dim=1)
                cur_act = torch.cat([cur_act, next_act], dim=1)

        return torch.cat(preds, dim=1)

    def score_reachability(self, src_emb, goal_emb, horizon):
        head = getattr(self, "reachability_head", None)
        if head is None:
            raise RuntimeError("Reachability head is not available on this model.")
        return head(src_emb, goal_emb, horizon)

    def decode_grounding(self, emb):
        head = getattr(self, "grounding_head", None)
        if head is None:
            raise RuntimeError("Grounding head is not available on this model.")
        flat = emb.reshape(-1, emb.size(-1))
        pred = head(flat)
        return pred.reshape(*emb.shape[:-1], pred.size(-1))

    def score_temporal_distance(self, src_emb, goal_emb):
        head = getattr(self, "temporal_distance_head", None)
        if head is None:
            raise RuntimeError("Temporal distance head is not available on this model.")
        return head(src_emb, goal_emb)

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"]

        while goal_emb.ndim < pred_emb.ndim:
            goal_emb = goal_emb.unsqueeze(1)

        goal_last = goal_emb[..., -1:, :]
        scored_pred = pred_emb[..., 1:, :] if pred_emb.size(-2) > 1 else pred_emb
        step_cost = F.mse_loss(
            scored_pred,
            goal_last.expand_as(scored_pred).detach(),
            reduction="none",
        ).sum(dim=-1)

        reduce = getattr(self, "goal_cost_reduce", "terminal")
        if reduce == "min":
            base_cost = step_cost.min(dim=-1).values
        elif reduce == "softmin":
            temperature = max(float(getattr(self, "goal_cost_softmin_temperature", 1.0)), 1e-6)
            base_cost = -temperature * torch.logsumexp(-step_cost / temperature, dim=-1)
        elif reduce == "mean":
            base_cost = step_cost.mean(dim=-1)
        else:
            base_cost = step_cost[..., -1]

        cost = base_cost
        head = getattr(self, "reachability_head", None)
        use_reachability = getattr(self, "use_reachability_cost", False)
        if head is not None and use_reachability and pred_emb.size(-2) > 1:
            future_pred = pred_emb[..., 1:, :]
            goal_future = goal_last.expand_as(future_pred)
            steps = future_pred.size(-2)
            rem_horizon = torch.arange(
                steps,
                0,
                -1,
                device=future_pred.device,
                dtype=torch.long,
            )
            rem_shape = [1] * (future_pred.ndim - 2) + [steps]
            rem_horizon = rem_horizon.view(*rem_shape).expand(*future_pred.shape[:-1])

            logits = self.score_reachability(
                future_pred.reshape(-1, future_pred.size(-1)),
                goal_future.reshape(-1, goal_future.size(-1)),
                rem_horizon.reshape(-1),
            )
            reach_prob = torch.sigmoid(logits).reshape(*future_pred.shape[:-1])
            best_reach = reach_prob.max(dim=-1).values

            reach_weight = getattr(self, "reachability_cost_weight", 0.75)
            latent_weight = getattr(self, "latent_cost_weight", 1.0)
            multiplier = (1.0 - reach_weight * best_reach).clamp_min(0.05)
            cost = latent_weight * base_cost * multiplier

        td_head = getattr(self, "temporal_distance_head", None)
        use_td = getattr(self, "use_temporal_distance_cost", False)
        if td_head is not None and use_td and pred_emb.size(-2) > 1:
            future_pred = pred_emb[..., 1:, :]
            goal_future = goal_last.expand_as(future_pred)
            td_logits = self.score_temporal_distance(
                future_pred.reshape(-1, future_pred.size(-1)),
                goal_future.reshape(-1, goal_future.size(-1)),
            ).reshape(*future_pred.shape[:-1], -1)
            td_probs = torch.softmax(td_logits, dim=-1)

            num_classes = td_logits.size(-1) - 1
            td_values = torch.arange(1, num_classes + 1, device=td_logits.device, dtype=td_probs.dtype)
            td_values = torch.cat(
                [td_values.new_tensor([num_classes + 1]), td_values],
                dim=0,
            )
            td_step_cost = (td_probs * td_values.view(*([1] * (td_probs.ndim - 1)), -1)).sum(dim=-1)

            td_reduce = getattr(self, "temporal_distance_cost_reduce", "min")
            if td_reduce == "mean":
                td_cost = td_step_cost.mean(dim=-1)
            elif td_reduce == "terminal":
                td_cost = td_step_cost[..., -1]
            else:
                td_cost = td_step_cost.min(dim=-1).values

            td_weight = getattr(self, "temporal_distance_cost_weight", 0.0)
            cost = cost + td_weight * td_cost

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_") and k in goal:
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action", None)
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)

        action_l2_weight = float(getattr(self, "action_l2_cost_weight", 0.0))
        if action_l2_weight > 0.0:
            action_l2 = action_candidates.square().mean(dim=tuple(range(2, action_candidates.ndim)))
            cost = cost + action_l2_weight * action_l2

        action_smooth_weight = float(getattr(self, "action_smooth_cost_weight", 0.0))
        if action_smooth_weight > 0.0 and action_candidates.size(2) > 1:
            diffs = action_candidates[:, :, 1:] - action_candidates[:, :, :-1]
            action_smooth = diffs.square().mean(dim=tuple(range(2, diffs.ndim)))
            cost = cost + action_smooth_weight * action_smooth
        
        return cost
