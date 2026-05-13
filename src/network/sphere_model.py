import random
import warnings
from typing import Dict, Union, List, Optional

import einops
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum, Tensor
from trimesh import Trimesh

from network.sphere_PSA import AdaptiveSphericalAttention
from models.heads.sdf_head import SDFBoundaryHead
from trimesh_utils import get_icosphere, IcoSphereRef, asSpherical


class MLP(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, out_dim=32, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act_layer(),
        )
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, out_dim))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x


#########################################
# Downsample Block
class MaxDownsample(nn.Module):
    def __init__(self, in_rank: int, out_rank: int, ref: IcoSphereRef):
        super().__init__()
        assert ref.node_type == "face"
        downscale = in_rank - out_rank
        self.swap_dims = True  # maxpool does operation on last dimension
        self.pool = nn.MaxPool1d(4 ** downscale, 4 ** downscale)

    def forward(self, x: Tensor):
        if self.swap_dims:
            x = einops.rearrange(x, "n c d -> n d c")
        x = self.pool(x)
        if self.swap_dims:
            x = einops.rearrange(x, "n d c -> n c d")
        return x


class AvgDownsample(nn.Module):
    def __init__(self, in_rank: int, out_rank: int, ref: IcoSphereRef):
        super().__init__()
        assert ref.node_type == "face"
        downscale = in_rank - out_rank
        self.swap_dims = True  # maxpool does operation on last dimension
        self.pool = nn.AvgPool1d(4 ** downscale, 4 ** downscale)

    def forward(self, x: Tensor):
        if self.swap_dims:
            x = einops.rearrange(x, "n c d -> n d c")
        x = self.pool(x)
        if self.swap_dims:
            x = einops.rearrange(x, "n d c -> n c d")
        return x


def _chunked_argmax_similarity(
    in_normals: np.ndarray,
    out_normals: np.ndarray,
    chunk_size: int = 256,
) -> List[int]:
    in_normals = in_normals.astype(np.float32, copy=False)
    out_normals = out_normals.astype(np.float32, copy=False)
    indices: List[int] = []
    for start in range(0, out_normals.shape[0], chunk_size):
        out_chunk = out_normals[start:start + chunk_size]
        sim = in_normals @ out_chunk.T
        indices.extend(sim.argmax(axis=0).tolist())
    return indices


def _exact_vertex_parent_indices(in_normals: np.ndarray, out_normals: np.ndarray) -> List[int]:
    out_count = out_normals.shape[0]
    if out_count <= in_normals.shape[0] and np.allclose(in_normals[:out_count], out_normals, atol=1e-8):
        return list(range(out_count))

    rounded_to_idx = {
        tuple(np.round(v, decimals=8)): idx
        for idx, v in enumerate(in_normals)
    }
    indices = []
    for v in out_normals:
        key = tuple(np.round(v, decimals=8))
        idx = rounded_to_idx.get(key)
        if idx is None:
            raise RuntimeError("Failed to build exact rank downsampling map for vertex icosphere.")
        indices.append(idx)
    return indices


class CenterDownsample(nn.Module):
    def __init__(self, in_rank: int, out_rank: int, ref: IcoSphereRef):
        super().__init__()

        self.downscale = in_rank - out_rank

        in_normals = ref.get_normals(in_rank)
        out_normals = ref.get_normals(out_rank)

        if ref.node_type == "vertex":
            center_idx = _exact_vertex_parent_indices(in_normals, out_normals)
        else:
            center_idx = _chunked_argmax_similarity(in_normals, out_normals)

        assert len(center_idx) == out_normals.shape[0],  f"{len(center_idx)} == {out_normals.shape[0]}"
        self.center_idx = center_idx

    def forward(self, x: Tensor):
        return x[:, self.center_idx, :]


#########################################
# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_rank: int, out_rank: int, ref: IcoSphereRef):
        super().__init__()
        self.upscale = out_rank - in_rank
        self.unpool = lambda x: einops.repeat(x, "n d c -> n (d k) c", k=4**self.upscale)

    def forward(self, x: Tensor):
        x = self.unpool(x)
        return x


class NearestUpsample(nn.Module):
    def __init__(self, in_rank: int, out_rank: int, ref: IcoSphereRef):
        super().__init__()

        self.upscale = out_rank - in_rank

        in_normals = ref.get_normals(in_rank)
        out_normals = ref.get_normals(out_rank)

        if out_rank < 7:
            cosine_similarity = in_normals @ out_normals.T
            center_idx = cosine_similarity.argmax(0).tolist()
        else:
            warnings.warn("BAD NearestUpsample")
            center_idx = [random.choice(range(in_normals.shape[0])) for _ in range(out_normals.shape[0])]

        assert len(center_idx) == out_normals.shape[0]
        self.center_idx = center_idx

    def forward(self, x: Tensor):
        return x[:, self.center_idx]


class InterpolateUpsample(nn.Module):
    def __init__(self, in_rank: int, out_rank: int, ref: IcoSphereRef):
        super().__init__()
        assert ref.node_type == "vertex"

        self.upscale = out_rank - in_rank
        assert self.upscale == 1

        in_ico = ref.get_icosphere(in_rank, refine=True)
        out_ico = ref.get_icosphere(out_rank, refine=True)

        in_size = in_ico.vertices.shape[0]
        out_size = out_ico.vertices.shape[0]
        self.left_idx = list(range(out_size))
        self.right_idx = list(range(out_size))

        for i in range(in_size, out_size):
            indices = [_ for _ in out_ico.vertex_neighbors[i] if _ < in_size]
            self.left_idx[i], self.right_idx[i] = indices

    def forward(self, x: Tensor):
        return (x[:, self.left_idx] + x[:, self.right_idx]) / 2


#########################################
# I/O Projections
class InputProj(nn.Module):
    def __init__(self, in_channel, out_channel, *, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channel, out_channel),
        )
        if act_layer is not None:
            self.proj.add_module(str(len(self.proj)), act_layer())
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class OutputProj(nn.Module):
    def __init__(self, in_channel, out_channel, *, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channel, out_channel),
        )
        if act_layer is not None:
            self.proj.add_module(str(len(self.proj)), act_layer())
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


#########################################
class AdaSpABlock(nn.Module):
    def __init__(
            self, *,
            rank: int,
            icosphere_ref: IcoSphereRef,
            dim, num_heads, d_head_coef, win_size_coef,
            mlp_ratio=4.,
            qkv_bias=True, qk_scale=None,
            attn_drop=0., attn_out_drop=0., mlp_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            geodesic_bias_mode: str = "mlp",
            geodesic_bias_scale: float = 1.0,
            contextual_ambiguity_aware_geodesic_bias_s_min: float = 0.8,
            contextual_ambiguity_aware_geodesic_bias_s_max: float = 1.6,
    ):
        super().__init__()

        self.rank = rank
        self.icosphere_ref = icosphere_ref

        self.dim = dim
        self.num_heads = num_heads
        self.win_size_coef = win_size_coef

        self.mlp_ratio = mlp_ratio

        self.attn = AdaptiveSphericalAttention(
            rank=rank,
            icosphere_ref=icosphere_ref,
            win_size_coef=win_size_coef,
            num_heads=num_heads,
            d_head_coef=d_head_coef,
            d_model=dim,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            out_drop=attn_out_drop,
            geodesic_bias_mode=geodesic_bias_mode,
            geodesic_bias_scale=geodesic_bias_scale,
            contextual_ambiguity_aware_geodesic_bias_s_min=(
                contextual_ambiguity_aware_geodesic_bias_s_min
            ),
            contextual_ambiguity_aware_geodesic_bias_s_max=(
                contextual_ambiguity_aware_geodesic_bias_s_max
            ),
        )

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.drop_path = DropPath(drop_path)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, dim, act_layer=act_layer, drop=mlp_drop)

    def extra_repr(self) -> str:
        return f"!!rank={self.rank}, dim={self.dim}, num_heads={self.num_heads}, win_size={self.win_size_coef}, mlp_ratio={self.mlp_ratio}!!"

    def forward(self, x: Tensor):
        bus = x

        # ATTN
        x_ = self.norm1(bus)
        x_, aux = self.attn(x=x_)
        bus = bus + self.drop_path(x_)

        # FFN
        x_ = self.norm2(bus)
        x_ = self.mlp(x_)
        bus = bus + self.drop_path(x_)
        return bus, aux


########### AdaSpA Stage ################
class AdaSpAStage(nn.Module):
    def __init__(
            self, *,
            rank: int,
            icosphere_ref: IcoSphereRef,
            dim,
            num_blocks,
            num_heads,
            d_head_coef,
            win_size_coef,
            mlp_ratio=4.,
            qkv_bias=True, qk_scale=None,
            attn_drop: float = 0., attn_out_drop: float = 0.,
            mlp_drop: float = 0., drop_path: Union[List[float], float] = 0.1,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            geodesic_bias_mode: str = "mlp",
            geodesic_bias_scale: float = 1.0,
            contextual_ambiguity_aware_geodesic_bias_s_min: float = 0.8,
            contextual_ambiguity_aware_geodesic_bias_s_max: float = 1.6,
    ):

        super().__init__()
        self.rank = rank
        self.icosphere_ref = icosphere_ref

        self.dim = dim
        self.num_blocks = num_blocks
        self.checkpoint_training = True

        # build blocks
        self.blocks = nn.ModuleList([
            AdaSpABlock(rank=rank, icosphere_ref=icosphere_ref,
                                  dim=dim, num_heads=num_heads, d_head_coef=d_head_coef,
                                  win_size_coef=win_size_coef,
                                  mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  attn_drop=attn_drop, attn_out_drop=attn_out_drop, mlp_drop=mlp_drop,
                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                  act_layer=act_layer, norm_layer=norm_layer,
                                  geodesic_bias_mode=geodesic_bias_mode,
                                  geodesic_bias_scale=geodesic_bias_scale,
                                  contextual_ambiguity_aware_geodesic_bias_s_min=(
                                      contextual_ambiguity_aware_geodesic_bias_s_min
                                  ),
                                  contextual_ambiguity_aware_geodesic_bias_s_max=(
                                      contextual_ambiguity_aware_geodesic_bias_s_max
                                  ),
                                  )
            for i in range(num_blocks)])

    def extra_repr(self) -> str:
        return f"!!rank={self.rank}, dim={self.dim}, num_blocks={self.num_blocks}!!"

    def forward(self, x):
        aux_vals = []

        for blk in self.blocks:
            x, aux = checkpoint.checkpoint(blk, x, use_reentrant=False) if self.checkpoint_training and self.training else blk(x)
            aux_vals.append(aux["contextual_ambiguity_mean"])

        return x, {"contextual_ambiguity_mean": torch.stack(aux_vals).mean()}


########### AdapToPASS ################
class AdapToPASS(nn.Module):
    def __init__(
            self,
            img_rank: int,
            node_type: str,
            in_channels=3,
            out_channels=1,
            embed_dim=32,
            num_scales=4,
            in_scale_factor: int = 2,
            enc_num_layers=(2, 2, 2, 2),
            bottleneck_num_layers=2,
            dec_num_layers=(2, 2, 2, 2),
            d_head_coef: int = 1,
            enc_num_heads=(2, 4, 8, 16),
            bottleneck_num_heads=None,
            dec_num_heads=(16, 16, 8, 4),
            win_size_coef: int = 1,
            mlp_ratio=4., qkv_bias=True, qk_scale=None,
            attn_drop_rate=0., attn_out_drop_rate=0., drop_rate=0., drop_path_rate=0., pos_drop_rate=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            downsample: str = "center",
            upsample: str = "nearest",
            geodesic_bias_mode: str = "mlp",
            geodesic_bias_scale: float = 1.0,
            acuity_stream_layers: int = 1,
            acuity_stream_heads: int = 2,
            acuity_stream_dim_ratio: int = 2,
            acuity_gate_init: float = -2.0,
            contextual_ambiguity_aware_geodesic_bias_s_min: float = 0.8,
            contextual_ambiguity_aware_geodesic_bias_s_max: float = 1.6,
    ):

        super().__init__()

        enc_num_heads = enc_num_heads or (1, 2, 4, 8, 16, 16)
        dec_num_heads = dec_num_heads or (16, 16, 16, 8, 4, 2)

        if isinstance(enc_num_layers, int):
            enc_num_layers = [enc_num_layers] * num_scales
        if isinstance(dec_num_layers, int):
            dec_num_layers = [dec_num_layers] * num_scales

        enc_num_layers = enc_num_layers[:num_scales]
        enc_num_heads = enc_num_heads[:num_scales]
        dec_num_layers = dec_num_layers[len(dec_num_layers)-num_scales:]
        dec_num_heads = dec_num_heads[len(dec_num_layers)-num_scales:]

        self.img_rank = img_rank
        self.lateral_rank = lateral_rank = img_rank - int(math.log2(in_scale_factor))
        self.embed_dim = embed_dim
        self.num_lateral_encoder_stages = len(enc_num_layers)
        self.num_lateral_decoder_stages = len(dec_num_layers)

        self.mlp_ratio = mlp_ratio
        self.win_size_coef = win_size_coef

        self.contextual_ambiguity_aware_geodesic_bias_s_min = float(
            contextual_ambiguity_aware_geodesic_bias_s_min
        )
        self.contextual_ambiguity_aware_geodesic_bias_s_max = float(
            contextual_ambiguity_aware_geodesic_bias_s_max
        )
        self.has_acuity_stream = bool(in_scale_factor > 1)
        self.acuity_stream_layers = int(acuity_stream_layers)
        self.acuity_stream_heads = int(acuity_stream_heads)
        self.acuity_stream_dim = max(8, embed_dim // max(1, int(acuity_stream_dim_ratio)))

        # Create Trimesh here for all ranks
        print("Generating sphere refs")
        self.icosphere_ref = IcoSphereRef(node_type=node_type)

        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        # Drop-path schedule
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_num_layers))]
        bottleneck_dpr = [drop_path_rate] * bottleneck_num_layers
        dec_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_num_layers))][::-1]

        self.lateral_input_proj = InputProj(in_channel=in_channels, out_channel=embed_dim, act_layer=nn.GELU)
        self.lateral_seg_head = OutputProj(in_channel=embed_dim, out_channel=out_channels)
        self.lateral_sdf_head = SDFBoundaryHead(embed_dim, out_channels)
        self.acuity_input_proj = None
        self.acuity_stream = None
        self.acuity_to_lateral = None
        self.acuity_seg_head = None
        self.acuity_sdf_head = None
        self.acuity_to_lateral_gate = None
        self.acuity_seg_gate = None
        self.acuity_sdf_gate = None

        if self.has_acuity_stream:
            # Keep a lightweight full-rank acuity stream alive and fuse it into the lateral trunk.
            self.acuity_input_proj = InputProj(in_channel=in_channels, out_channel=self.acuity_stream_dim, act_layer=nn.GELU)
            self.acuity_stream = AdaSpAStage(
                rank=img_rank,
                icosphere_ref=self.icosphere_ref,
                dim=self.acuity_stream_dim,
                num_blocks=self.acuity_stream_layers,
                num_heads=self.acuity_stream_heads,
                d_head_coef=d_head_coef,
                win_size_coef=win_size_coef,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop_rate,
                attn_out_drop=attn_out_drop_rate,
                mlp_drop=drop_rate,
                drop_path=0.0,
                act_layer=act_layer,
                norm_layer=norm_layer,
                geodesic_bias_mode=geodesic_bias_mode,
                geodesic_bias_scale=geodesic_bias_scale,
                contextual_ambiguity_aware_geodesic_bias_s_min=(
                    contextual_ambiguity_aware_geodesic_bias_s_min
                ),
                contextual_ambiguity_aware_geodesic_bias_s_max=(
                    contextual_ambiguity_aware_geodesic_bias_s_max
                ),
            )
            self.acuity_to_lateral = nn.Sequential(
                CenterDownsample(img_rank, lateral_rank, ref=self.icosphere_ref),
                norm_layer(self.acuity_stream_dim),
                nn.Linear(self.acuity_stream_dim, embed_dim),
            )
            self.acuity_seg_head = OutputProj(in_channel=self.acuity_stream_dim, out_channel=out_channels)
            self.acuity_sdf_head = SDFBoundaryHead(self.acuity_stream_dim, out_channels)
            self.acuity_to_lateral_gate = nn.Parameter(torch.tensor(acuity_gate_init, dtype=torch.float32))
            self.acuity_seg_gate = nn.Parameter(torch.tensor(acuity_gate_init, dtype=torch.float32))
            self.acuity_sdf_gate = nn.Parameter(torch.tensor(acuity_gate_init, dtype=torch.float32))

        if in_scale_factor > 1:
            self.lateral_input_proj = nn.Sequential(
                CenterDownsample(img_rank, lateral_rank, ref=self.icosphere_ref),
                self.lateral_input_proj,
            )
            self.lateral_seg_head = nn.Sequential(
                InterpolateUpsample(lateral_rank, img_rank, ref=self.icosphere_ref),
                self.lateral_seg_head,
            )
            self.lateral_sdf_head = nn.Sequential(
                InterpolateUpsample(lateral_rank, img_rank, ref=self.icosphere_ref),
                self.lateral_sdf_head,
            )

        downsample_layer = {
            "max": MaxDownsample,
            "avg": AvgDownsample,
            "center": CenterDownsample,
        }[downsample]

        upsample_layer = {
            "nearest": NearestUpsample,
            "interpolate": InterpolateUpsample,
        }[upsample]

        # Lateral encoder
        print("Building lateral encoder")
        self.lateral_encoder_stages = nn.ModuleList()
        self.lateral_downsample_blocks = nn.ModuleList()
        for i in range(self.num_lateral_encoder_stages):
            self.lateral_encoder_stages.append(
                nn.Sequential(
                    AdaSpAStage(
                        rank=lateral_rank-i,
                        icosphere_ref=self.icosphere_ref,
                        dim=embed_dim * (2 ** i),
                        num_blocks=enc_num_layers[i],
                        num_heads=enc_num_heads[i],
                        d_head_coef=d_head_coef,
                        win_size_coef=win_size_coef,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        attn_drop=attn_drop_rate, attn_out_drop=attn_out_drop_rate, mlp_drop=drop_rate,
                        drop_path=enc_dpr[int(sum(enc_num_layers[:i])):int(sum(enc_num_layers[:(i+1)]))],
                        act_layer=act_layer, norm_layer=norm_layer,
                        geodesic_bias_mode=geodesic_bias_mode,
                        geodesic_bias_scale=geodesic_bias_scale,
                        contextual_ambiguity_aware_geodesic_bias_s_min=(
                            self.contextual_ambiguity_aware_geodesic_bias_s_min
                        ),
                        contextual_ambiguity_aware_geodesic_bias_s_max=(
                            self.contextual_ambiguity_aware_geodesic_bias_s_max
                        ),
                    ),
                )
            )

            self.lateral_downsample_blocks.append(
                nn.Sequential(
                    downsample_layer(lateral_rank-i, lateral_rank-i-1, self.icosphere_ref),
                    norm_layer(embed_dim * (2 ** i)),
                    nn.Linear(in_features=embed_dim * (2 ** i), out_features=embed_dim * (2 ** i) * 2),
                    # nn.GELU(),
                    # norm_layer(embed_dim * (2 ** i) * 2),
                )
            )

        # Lateral bottleneck
        num_encoder_stages = self.num_lateral_encoder_stages
        self.lateral_bottleneck = AdaSpAStage(
            rank=lateral_rank-num_encoder_stages,
            icosphere_ref=self.icosphere_ref,
            dim=embed_dim * (2 ** num_encoder_stages),
            num_blocks=bottleneck_num_layers,
            num_heads=bottleneck_num_heads or dec_num_heads[0],
            d_head_coef=d_head_coef,
            win_size_coef=win_size_coef,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            mlp_drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=bottleneck_dpr,
            act_layer=act_layer, norm_layer=norm_layer,
            geodesic_bias_mode=geodesic_bias_mode,
            geodesic_bias_scale=geodesic_bias_scale,
            contextual_ambiguity_aware_geodesic_bias_s_min=(
                self.contextual_ambiguity_aware_geodesic_bias_s_min
            ),
            contextual_ambiguity_aware_geodesic_bias_s_max=(
                self.contextual_ambiguity_aware_geodesic_bias_s_max
            ),
        )

        # Lateral decoder
        print("Building lateral decoder")
        self.lateral_decoder_stages = nn.ModuleList()
        self.lateral_decoder_upsample_norm_layers = nn.ModuleList()
        self.lateral_decoder_skip_norm_layers = nn.ModuleList()
        self.lateral_upsample_blocks = nn.ModuleList()
        for i in range(self.num_lateral_decoder_stages):
            reverse_i = num_encoder_stages - i - 1

            self.lateral_upsample_blocks.append(
                nn.Sequential(
                    norm_layer(embed_dim * (2 ** reverse_i) * 2),
                    nn.Linear(in_features=embed_dim * (2 ** reverse_i) * 2, out_features=embed_dim * (2 ** reverse_i)),
                    # nn.GELU(),
                    # norm_layer(embed_dim * (2 ** reverse_i)),
                    upsample_layer(lateral_rank-reverse_i-1, lateral_rank-reverse_i, ref=self.icosphere_ref),
                )
            )

            self.lateral_decoder_upsample_norm_layers.append(
                norm_layer(embed_dim * (2 ** reverse_i)),
            )
            self.lateral_decoder_skip_norm_layers.append(
                norm_layer(embed_dim * (2 ** reverse_i)),
            )
            self.lateral_decoder_stages.append(
                nn.Sequential(
                    nn.Linear(in_features=embed_dim * (2 ** reverse_i) * 2, out_features=embed_dim * (2 ** reverse_i)),
                    # nn.GELU(),
                    # norm_layer(embed_dim * (2 ** reverse_i)),
                    AdaSpAStage(
                        rank=lateral_rank-reverse_i,
                        icosphere_ref=self.icosphere_ref,
                        dim=embed_dim * (2 ** reverse_i),
                        num_blocks=dec_num_layers[i],
                        num_heads=dec_num_heads[i],
                        d_head_coef=d_head_coef,
                        win_size_coef=win_size_coef,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        mlp_drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dec_dpr[int(sum(dec_num_layers[:i])):int(sum(dec_num_layers[:(i + 1)]))],
                        act_layer=act_layer, norm_layer=norm_layer,
                        geodesic_bias_mode=geodesic_bias_mode,
                        geodesic_bias_scale=geodesic_bias_scale,
                        contextual_ambiguity_aware_geodesic_bias_s_min=(
                            self.contextual_ambiguity_aware_geodesic_bias_s_min
                        ),
                        contextual_ambiguity_aware_geodesic_bias_s_max=(
                            self.contextual_ambiguity_aware_geodesic_bias_s_max
                        ),
                    )
                )
            )

        print("Initializing weights")
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return set()

    def extra_repr(self) -> str:
        return f"!!img_rank={self.img_rank}, embed_dim={self.embed_dim}, win_size={self.win_size_coef}!!"

    def forward(self, x):
        acuity_feat = None
        if self.has_acuity_stream:
            acuity_feat, acuity_aux = self.acuity_stream(self.acuity_input_proj(x))

        lateral_features = self.lateral_input_proj(x)
        contextual_ambiguity_means = []
        if acuity_feat is not None:
            contextual_ambiguity_means.append(acuity_aux["contextual_ambiguity_mean"])

        if acuity_feat is not None:
            lateral_features = (
                lateral_features
                + torch.sigmoid(self.acuity_to_lateral_gate) * self.acuity_to_lateral(acuity_feat)
            )

        lateral_features = self.pos_drop(lateral_features)

        # Lateral encoder
        lateral_encoder_features = []
        for i in range(len(self.lateral_encoder_stages)):
            encoder_features, aux = self.lateral_encoder_stages[i](lateral_features)
            contextual_ambiguity_means.append(aux["contextual_ambiguity_mean"])
            lateral_encoder_features.append(encoder_features)
            lateral_features = self.lateral_downsample_blocks[i](encoder_features)

        # Lateral bottleneck
        lateral_features, aux = self.lateral_bottleneck(lateral_features)
        contextual_ambiguity_means.append(aux["contextual_ambiguity_mean"])

        # Lateral decoder
        for i in range(len(self.lateral_decoder_stages)):
            lateral_features = self.lateral_upsample_blocks[i](lateral_features)
            lateral_features = torch.cat(
                [
                    self.lateral_decoder_upsample_norm_layers[i](lateral_features),
                    self.lateral_decoder_skip_norm_layers[i](
                        lateral_encoder_features[self.num_lateral_decoder_stages - 1 - i]
                    ),
                ],
                dim=-1,
            )
            lateral_features, aux = self.lateral_decoder_stages[i](lateral_features)
            contextual_ambiguity_means.append(aux["contextual_ambiguity_mean"])

        # Final heads
        logits = self.lateral_seg_head(lateral_features)
        sdf_pred = self.lateral_sdf_head(lateral_features)
        if acuity_feat is not None:
            logits = logits + torch.sigmoid(self.acuity_seg_gate) * self.acuity_seg_head(acuity_feat)
            sdf_pred = sdf_pred + torch.sigmoid(self.acuity_sdf_gate) * self.acuity_sdf_head(acuity_feat)

        return logits, sdf_pred, {"contextual_ambiguity_mean": torch.stack(contextual_ambiguity_means).mean()}
