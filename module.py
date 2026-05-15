import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        self.last_stats = {}
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def _statistic(self, proj):
        proj = proj.float()
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        self.last_stats = {}
        return self._statistic(proj)

    def get_stats(self):
        return self.last_stats


class AdaptiveSIGReg(SIGReg):
    """Adaptive low-rank Gaussian regularizer using EMA covariance whitening."""

    def __init__(
        self,
        knots=17,
        num_proj=1024,
        momentum=0.95,
        energy=0.98,
        min_rank=4,
        max_rank=None,
        rank_scale=1.0,
        var_floor=1e-4,
        mean_weight=1.0,
        tail_weight=0.05,
        isotropic_mix=0.0,
    ):
        super().__init__(knots=knots, num_proj=num_proj)
        self.momentum = momentum
        self.energy = energy
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.rank_scale = rank_scale
        self.var_floor = var_floor
        self.mean_weight = mean_weight
        self.tail_weight = tail_weight
        self.isotropic_mix = isotropic_mix
        self.register_buffer("ema_cov", torch.empty(0))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    def _update_cov(self, cov):
        cov = cov.detach()
        if self.ema_cov.numel() != cov.numel():
            self.ema_cov = cov.clone()
            self.num_updates.zero_()

        if self.num_updates.item() == 0:
            self.ema_cov.copy_(cov)
        else:
            self.ema_cov.lerp_(cov, 1.0 - self.momentum)
        self.num_updates.add_(1)

    def _compute_rank(self, evals, dim):
        evals = evals.clamp_min(1e-8)
        total = evals.sum()
        probs = evals / total
        eff_rank = torch.exp(-(probs * probs.log()).sum())
        cum_energy = probs.cumsum(0)
        energy_rank = (
            torch.searchsorted(cum_energy, cum_energy.new_tensor(self.energy)).item() + 1
        )
        scaled_rank = int(round(self.rank_scale * eff_rank.item()))
        max_rank = dim if self.max_rank is None else min(self.max_rank, dim)
        rank = max(self.min_rank, energy_rank, scaled_rank)
        rank = max(1, min(rank, max_rank))
        return rank, eff_rank

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        proj = proj.float()
        _, _, dim = proj.shape
        x = rearrange(proj, "t b d -> (t b) d")
        mean = x.mean(dim=0, keepdim=True)
        centered = x - mean

        cov = centered.t() @ centered / max(centered.size(0) - 1, 1)
        self._update_cov(cov)

        with torch.no_grad():
            cov_ref = cov if self.num_updates.item() == 1 else self.ema_cov
            eye = torch.eye(dim, device=proj.device, dtype=proj.dtype)
            evals, evecs = torch.linalg.eigh(cov_ref + 1e-6 * eye)
            rank, eff_rank = self._compute_rank(evals, dim)
            tail = evals[:-rank] if rank < dim else evals.new_zeros(0)
            tail_energy = tail.sum().item() if tail.numel() > 0 else 0.0
            tail_mean = tail.mean().item() if tail.numel() > 0 else 0.0
            keep_evals = evals[-rank:].clamp_min(self.var_floor)
            keep_vecs = evecs[:, -rank:]
            whiten = keep_vecs @ torch.diag(keep_evals.rsqrt())
            if self.isotropic_mix > 0.0:
                mean_var = keep_evals.mean()
                iso = keep_vecs @ torch.diag(torch.full_like(keep_evals, mean_var.rsqrt()))
                whiten = (1.0 - self.isotropic_mix) * whiten + self.isotropic_mix * iso
            mean_norm = mean.norm().item()

        centered = centered @ whiten
        centered = rearrange(centered, "(t b) r -> t b r", t=proj.size(0), b=proj.size(1))
        gaussian_penalty = self._statistic(centered)
        mean_penalty = mean.square().mean()

        loss = self.mean_weight * mean_penalty + self.tail_weight * tail_mean + gaussian_penalty
        self.last_stats = {
            "mean_norm": mean.new_tensor(mean_norm),
            "eff_rank": mean.new_tensor(eff_rank.item()),
            "rank": mean.new_tensor(float(rank)),
            "tail_energy": mean.new_tensor(tail_energy),
            "tail_mean": mean.new_tensor(tail_mean),
            "gaussian_penalty": gaussian_penalty.detach(),
        }
        return loss


class ReachabilityHead(nn.Module):
    """Budget-conditioned reachability classifier over latent pairs."""

    def __init__(
        self,
        embed_dim,
        hidden_dim=512,
        max_horizon=5,
        horizon_dim=32,
    ):
        super().__init__()
        self.max_horizon = max_horizon
        self.horizon_emb = nn.Embedding(max_horizon + 1, horizon_dim)
        input_dim = 4 * embed_dim + horizon_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, src, goal, horizon):
        horizon = horizon.long().clamp(0, self.max_horizon)
        horizon_feat = self.horizon_emb(horizon)
        feat = torch.cat(
            [
                src,
                goal,
                (src - goal).abs(),
                src * goal,
                horizon_feat,
            ],
            dim=-1,
        )
        return self.net(feat).squeeze(-1)


class GroundingHead(nn.Module):
    """Regression head used to ground latent embeddings to low-dimensional state."""

    def __init__(self, embed_dim, output_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, emb):
        return self.net(emb)


class TemporalDistanceHead(nn.Module):
    """Predict exact temporal distance buckets between latent pairs."""

    def __init__(self, embed_dim, hidden_dim=512, max_horizon=5):
        super().__init__()
        self.max_horizon = max_horizon
        input_dim = 4 * embed_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_horizon + 1),
        )

    def forward(self, src, goal):
        feat = torch.cat(
            [
                src,
                goal,
                (src - goal).abs(),
                src * goal,
            ],
            dim=-1,
        )
        return self.net(feat)
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x
