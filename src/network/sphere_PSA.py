from typing import Optional

import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from network.position_encoding import SphereNeighborhood
from trimesh_utils import IcoSphereRef


class GeodesicBias(nn.Module):
    """
    Geometry-aware bias that depends only on geodesic distance between
    query/neighbor nodes on the icosphere. The distance tensor is
    precomputed from icosphere normals and the neighbor mapping.
    """

    def __init__(
            self,
            *,
            normals,
            idx: torch.Tensor,
            idx_mask: torch.Tensor,
            num_heads: int,
            mode: str = "mlp",
            hidden_dim: int = 16,
            scale_init: float = 1.0,
    ):
        super().__init__()
        self.mode = mode
        self.num_heads = num_heads

        # idx: [1, 1, D, K, 1] ; idx_mask: [1, 1, D, K]
        idx_ = idx.squeeze(0).squeeze(0).squeeze(-1)  # [D, K]
        idx_mask_ = idx_mask.squeeze(0).squeeze(0)  # [D, K]

        normals = torch.as_tensor(normals, dtype=torch.float32)
        query_normals = normals.unsqueeze(1)  # [D, 1, 3]
        neighbor_normals = normals[idx_]  # [D, K, 3]

        cos = (query_normals * neighbor_normals).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
        dist = torch.acos(cos)  # [D, K]
        dist = torch.where(idx_mask_, dist, torch.zeros_like(dist))
        # shape to broadcast with batch/heads: [1, 1, D, K, 1]
        self.register_buffer("dists", dist.unsqueeze(0).unsqueeze(0).unsqueeze(-1), persistent=False)
        self.register_buffer("idx_mask", idx_mask, persistent=False)

        if self.mode == "mlp":
            self.bias_net = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_heads),
            )
        elif self.mode == "rbf":
            # simple RBF with learnable bandwidth
            self.log_gamma = nn.Parameter(torch.tensor(0.0))
            self.head_proj = nn.Linear(1, num_heads)
        else:
            raise ValueError(f"Unsupported geodesic_bias_mode {mode}")

        self.geo_scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))

        # start near-zero bias to preserve baseline behavior
        if self.mode == "mlp":
            nn.init.zeros_(self.bias_net[-1].weight)
            nn.init.zeros_(self.bias_net[-1].bias)
        else:
            nn.init.zeros_(self.head_proj.weight)
            nn.init.zeros_(self.head_proj.bias)

    def _compute_bias(self, dist: torch.Tensor) -> torch.Tensor:
        if self.mode == "mlp":
            return self.bias_net(dist)
        # RBF fallback
        gamma = torch.exp(self.log_gamma)
        feat = torch.exp(-gamma * dist ** 2)
        return self.head_proj(feat)

    def forward(self, batch_size: int, device: torch.device, dist_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns: bias tensor shaped [batch, num_heads, D, K]
        dist_override: optional pre-scaled distances shaped [B, 1, D, K] or [B, 1, D, K, 1]
        """
        if dist_override is None:
            dist = self.dists.to(device).expand(batch_size, -1, -1, -1, -1)  # [B,1,D,K,1]
        else:
            dist = dist_override
            if dist.dim() == 4:  # [B,1,D,K]
                dist = dist.unsqueeze(-1)
            dist = dist.to(device)

        bias = self._compute_bias(dist)  # [B,1,D,K,H]
        bias = bias.permute(0, 4, 2, 3, 1).squeeze(-1)  # [N, H, D, K]
        bias = bias * self.geo_scale
        # mask invalid keys to zero (attn logits will be -inf after mask)
        mask = self.idx_mask.to(device).expand(batch_size, -1, -1, -1)
        return torch.where(mask, bias, torch.zeros_like(bias))

    @torch.no_grad()
    def get_bias_matrix(self, device: torch.device) -> torch.Tensor:
        """Helper for tests: returns [num_heads, D, K] on given device."""
        bias = self.forward(batch_size=1, device=device)[0]
        return bias

    @torch.no_grad()
    def get_distances(self, device: torch.device) -> torch.Tensor:
        """Helper for tests: returns [D, K] geodesic distances."""
        return self.dists.to(device).squeeze(0).squeeze(0).squeeze(-1)


class AdaptiveSphericalAttention(nn.Module):
    LOGIT_SCALE_PRE_REL_BIAS: bool = True

    def __init__(
            self, *,
            rank: int,
            icosphere_ref: IcoSphereRef,
            win_size_coef,
            num_heads,
            d_model,
            d_head_coef,
            qkv_bias,
            attn_drop=0.,
            out_drop=0.,
            geodesic_bias_mode: str = "mlp",
            geodesic_bias_scale: float = 1.0,
            contextual_ambiguity_aware_geodesic_bias_s_min: float = 0.8,
            contextual_ambiguity_aware_geodesic_bias_s_max: float = 1.6,
    ):
        """
        :param num_heads: number of self attention head
        :param d_model: dimension of model
        :param dropout:
        :param num_keys: number of keys
        """

        super().__init__()

        self.rank = rank
        self.icosphere_ref = icosphere_ref
        self.win_size_coef = win_size_coef

        self.neighborhood = SphereNeighborhood(
            rank,
            icosphere_ref,
            win_size_coef,
        )
        self.num_keys = self.neighborhood.num_keys

        # We assume d_v always equals d_k, d_q = d_k = d_v = d_m // h
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = (d_model // num_heads) * d_head_coef
        assert self.d_head == int(self.d_head)

        self.q_proj = nn.Linear(d_model, self.d_head * num_heads, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, self.d_head * num_heads, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, self.d_head * num_heads, bias=qkv_bias)

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        self.out_proj = nn.Linear(self.d_head * num_heads, d_model)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_drop = nn.Dropout(out_drop)

        self.geo_bias = GeodesicBias(
            normals=icosphere_ref.get_normals(rank),
            idx=self.neighborhood.idx,
            idx_mask=self.neighborhood.idx_mask,
            num_heads=num_heads,
            mode=geodesic_bias_mode,
            scale_init=geodesic_bias_scale,
        )

        self.contextual_ambiguity_aware_geodesic_bias_s_min = float(
            contextual_ambiguity_aware_geodesic_bias_s_min
        )
        self.contextual_ambiguity_aware_geodesic_bias_s_max = float(
            contextual_ambiguity_aware_geodesic_bias_s_max
        )
        hidden_dim = max(4, d_model // 4)
        self.contextual_ambiguity_layer_norm = nn.LayerNorm(d_model)
        self.contextual_ambiguity_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.constant_(self.contextual_ambiguity_mlp[-1].bias, -2.0)  # start with low contextual ambiguity

    def forward(
            self,
            x: Tensor,
            query_mask: Tensor = None,
            key_masks: Optional[Tensor] = None,

    ):
        """
        :param x: B, D, C
        :param query_mask:
        :param key_masks:
        :return:
        """
        N, D, C = x.shape
        H = self.num_heads
        K = self.num_keys
        C_H = self.d_head

        # q,k,v: [N, H, D, C//H]
        q = self.q_proj(x).view(N, D, H, C_H).permute(0,2,1,3)
        k = self.k_proj(x).view(N, D, H, C_H).permute(0,2,1,3)
        v = self.v_proj(x).view(N, D, H, C_H).permute(0,2,1,3)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # aligned: [N, H, D, K, C//H]
        # expanded_idx = self.idx[None, None, :, :, None].expand(N, H, -1, -1, C_H)
        # expanded_idx_mask = self.idx_mask[None, None, :, :].expand(N, H, -1, -1)
        expanded_idx = self.neighborhood.idx.expand(N, H, -1, -1, C_H)
        expanded_idx_mask = self.neighborhood.idx_mask.expand(N, H, -1, -1)
        expanded_k = k[:, :, :, None, :].expand(-1, -1, -1, K, -1)
        expanded_v = v[:, :, :, None, :].expand(-1, -1, -1, K, -1)

        aligned_k = torch.gather(expanded_k, dim=2, index=expanded_idx)
        aligned_v = torch.gather(expanded_v, dim=2, index=expanded_idx)

        # (q: [N, H, D, C_H] , aligned_k: [N, H, D, K, C_H]) -> attn: [N, H, D, K]
        attn = (q[:, :, :, None, :] * aligned_k).sum(-1)

        if self.LOGIT_SCALE_PRE_REL_BIAS:
            logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01, device=self.logit_scale.device))).exp()
            attn = attn * logit_scale

        aux = {}

        # Contextual-Ambiguity-Aware Geodesic Bias predicts one ambiguity score per query token.
        contextual_ambiguity_score = torch.sigmoid(
            self.contextual_ambiguity_mlp(self.contextual_ambiguity_layer_norm(x))
        )  # [N, D, 1]
        contextual_ambiguity_scale = (
            self.contextual_ambiguity_aware_geodesic_bias_s_min
            + (
                self.contextual_ambiguity_aware_geodesic_bias_s_max
                - self.contextual_ambiguity_aware_geodesic_bias_s_min
            )
            * contextual_ambiguity_score
        )  # [N, D, 1]
        base_dist = self.geo_bias.dists.to(x.device).expand(N, -1, -1, -1, -1)  # [N,1,D,K,1]
        scale = contextual_ambiguity_scale.unsqueeze(1).unsqueeze(-1)  # [N,1,D,1,1]
        dist_override = torch.clamp(base_dist / (scale + 1e-6), 0.0, math.pi)
        aux["contextual_ambiguity_mean"] = contextual_ambiguity_score.mean()

        geo_bias = self.geo_bias(batch_size=N, device=x.device, dist_override=dist_override)  # [N, H, D, K]
        attn = attn + geo_bias

        if not self.LOGIT_SCALE_PRE_REL_BIAS:
            logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
            attn = attn * logit_scale

        # mask for cases where number of keys < max number of keys
        attn = torch.masked_fill(attn, mask=~expanded_idx_mask, value=float('-inf'))

        # B, H, D, K
        attn = F.softmax(attn, dim=-1)

        # mask nan position
        if query_mask is not None:
            raise NotImplementedError("No support for query_mask")
            # B, H, W, 1, 1
            query_mask_ = query_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
            attn = torch.masked_fill(attn, query_mask_.expand_as(attn), 0.0)

        attn = self.attn_drop(attn)

        # (attn: [N, H, D, K] , aligned_v: [N, H, D, K, C_H]) -> out: [N, H, D, C_H]
        out = (attn.unsqueeze(-1) * aligned_v).sum(-2)
        out = einops.rearrange(out, "N H D C_H -> N D (H C_H)")
        out = self.out_proj(out)
        out = self.out_drop(out)
        return out, aux
