#!/usr/bin/env python3
"""
Single-image semantic segmentation inference for AdapToPASS.

Outputs:
- <out_dir>/<stem>_pred_labels.png
- <out_dir>/<stem>_overlay.png
"""

import argparse
import contextlib
import io
import os
import re
import shutil
import sys
import time
import warnings
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.layers is deprecated, please import via timm\.layers",
    category=FutureWarning,
)

from network.sphere_model import AdapToPASS
from trimesh_utils import IcoSphereRef, asSpherical


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_weights")
MASK_PATH = os.path.join(DATA_DIR, "stanford2d3d_mask.png")

COLOR_ENABLED = sys.stdout.isatty() and os.environ.get("TERM", "dumb") != "dumb"


def _style(text: str, *codes: str) -> str:
    if not COLOR_ENABLED or not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def _accent(text: str) -> str:
    return _style(text, "1", "38;5;45")


def _glow(text: str) -> str:
    return _style(text, "38;5;219")


def _soft(text: str) -> str:
    return _style(text, "38;5;110")


def _muted(text: str) -> str:
    return _style(text, "38;5;244")


def _term_width(default: int = 88) -> int:
    return max(72, min(108, shutil.get_terminal_size((default, 20)).columns))


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _vlen(text: str) -> int:
    return len(_ANSI_RE.sub("", text))


def _shorten(text: str, width: int) -> str:
    text = str(text)
    if _vlen(text) <= width:
        return text
    if width <= 8:
        return text[:width]
    keep = (width - 1) // 2
    return f"{text[:keep]}…{text[-(width - keep - 1):]}"


def _boxed(lines):
    width = _term_width()
    inner = width - 4
    top = _accent("┌" + "─" * (width - 2) + "┐")
    bottom = _accent("└" + "─" * (width - 2) + "┘")
    print(top)
    for line in lines:
        text = _shorten(line, inner)
        padding = " " * max(0, inner - _vlen(text))
        print(_accent("│ ") + text + padding + _accent(" │"))
    print(bottom)


def _emit_banner(image_path: str, output_dir: str, device: torch.device) -> None:
    _boxed(
        [
            _glow("AdapToPASS // spherical segmentation"),
            _soft(f"scene  {_shorten(os.path.basename(image_path), _term_width() - 16)}"),
            _soft(f"device {device}"),
            _soft(f"save   {_shorten(os.path.abspath(output_dir), _term_width() - 16)}"),
        ]
    )


def _emit_success(pred_png: str, overlay_png: str, elapsed_s: float) -> None:
    _boxed(
        [
            _glow("render complete"),
            _soft(f"labels  {_shorten(pred_png, _term_width() - 16)}"),
            _soft(f"overlay {_shorten(overlay_png, _term_width() - 16)}"),
            _soft(f"time    {elapsed_s:.2f}s"),
        ]
    )


def _emit_ascii_art() -> None:
    w = _term_width()

    def s(t, c): return _style(t, c) if COLOR_ENABLED else t
    B  = lambda t: s(t, "1;38;5;45")   # bright cyan borders
    P  = lambda t: s(t, "38;5;219")    # pink fill
    V  = lambda t: s(t, "38;5;183")    # violet fill
    G  = lambda t: s(t, "38;5;85")     # mint fill
    Y  = lambda t: s(t, "38;5;222")    # gold separator
    DM = lambda t: s(t, "38;5;238")    # very dim dots

    IW = 34
    mg = " " * max(0, (w - IW - 6) // 2)

    def row(fn, text):
        return mg + B("  │ ") + fn(text.center(IW)) + B(" │")

    title = mg + P(("◆  spherical segmentation complete  ◆").center(IW + 6))
    dots  = mg + DM("  · " + "·  " * 12)
    top   = mg + B("  ╭") + B("─" * (IW + 2)) + B("╮")
    hbar  = mg + B("  ╞") + Y("═" * (IW + 2)) + B("╡")
    bot   = mg + B("  ╰") + B("─" * (IW + 2)) + B("╯")

    print()
    print(title)
    print(dots)
    print(top)
    print(row(P,  "·  " * 11))
    print(row(V,  "·   A d a p T o P A S S   ·"))
    print(hbar)
    print(row(G,  "·   3 6 0 °   s p h e r e   ·"))
    print(row(P,  "·  " * 11))
    print(bot)
    print(dots)
    print()


@contextlib.contextmanager
def _suppress_internal_output():
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_colors(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected (K,3) color array at {path}, got {arr.shape}")
    return arr.astype(np.uint8)


def load_binary_mask(path: str, out_hw: Tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")

    out_h, out_w = out_hw
    if mask.shape != (out_h, out_w):
        mask = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def load_gt_ignore_mask(path: str, out_hw: Tuple[int, int], ignore_rgb: np.ndarray) -> np.ndarray:
    gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if gt is None:
        raise FileNotFoundError(f"Could not read GT: {path}")

    if gt.ndim != 3 or gt.shape[2] != 4:
        raise ValueError(f"Expected RGBA GT at {path}, got shape {gt.shape}")

    gt = cv2.cvtColor(gt, cv2.COLOR_BGRA2RGBA)
    out_h, out_w = out_hw
    if gt.shape[:2] != (out_h, out_w):
        gt = cv2.resize(gt, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    gt_rgb = gt[..., :3]
    gt_alpha = gt[..., 3] > 0
    ignore_rgb = np.asarray(ignore_rgb, dtype=np.uint8).reshape(1, 1, 3)
    keep = gt_alpha & np.any(gt_rgb != ignore_rgb, axis=2)
    return np.where(keep, 255, 0).astype(np.uint8)


def read_rgb(path: str) -> np.ndarray:
    rgb = cv2.imread(path, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def resolve_gt_path(image_path: str) -> Optional[str]:
    image_path = os.path.abspath(image_path)
    stem = os.path.basename(image_path)
    if os.sep + "rgb" + os.sep not in image_path or not stem.endswith("_rgb.png"):
        return None

    gt_path = image_path.replace(f"{os.sep}rgb{os.sep}", f"{os.sep}gt{os.sep}")
    gt_path = gt_path[:-len("_rgb.png")] + "_gt.png"
    if os.path.isfile(gt_path):
        return gt_path
    return None


def to_tensor_sphere_rgb_like_dataset(
    erp_rgb: np.ndarray,
    sphere_rank: int,
    node_type: str,
) -> Tuple[torch.Tensor, np.ndarray]:
    icos_ref = IcoSphereRef(node_type)
    normals = icos_ref.get_normals(rank=sphere_rank)
    rphitheta = asSpherical(normals)
    normals_wh = np.stack(
        (
            rphitheta[:, 2] / 180.0,
            rphitheta[:, 1] / 180.0 * 2 - 1,
        ),
        axis=1,
    ).astype(np.float32)

    sphere_grid = torch.from_numpy(normals_wh).reshape(1, -1, 1, 2)
    rgb_t = torch.from_numpy(erp_rgb).permute(2, 0, 1).float().unsqueeze(0)
    rgb_nodes = F.grid_sample(
        input=rgb_t,
        grid=sphere_grid,
        padding_mode="border",
        align_corners=False,
    )
    rgb_nodes = rgb_nodes.squeeze(3).permute(0, 2, 1).contiguous()
    if rgb_nodes.max() > 1.0:
        rgb_nodes = rgb_nodes / 255.0
    return rgb_nodes, normals_wh


def normalize_like_training(x: torch.Tensor, mean: float = 0.5, std: float = 0.225) -> torch.Tensor:
    mean_vec = torch.tensor([mean, mean, mean], dtype=x.dtype, device=x.device).view(1, 1, 3)
    std_vec = torch.tensor([std, std, std], dtype=x.dtype, device=x.device).view(1, 1, 3)
    return (x - mean_vec) / std_vec


def build_model(num_classes: int, args) -> AdapToPASS:
    return AdapToPASS(
        img_rank=args.img_rank,
        node_type=args.mode,
        in_channels=3,
        out_channels=num_classes,
        in_scale_factor=args.scale_factor,
        num_scales=args.num_scales,
        win_size_coef=args.win_size_coef,
        enc_num_layers=args.scale_layers,
        dec_num_layers=args.scale_layers,
        bottleneck_num_layers=args.scale_layers,
        d_head_coef=args.d_head_coef,
        enc_num_heads=args.enc_num_heads,
        bottleneck_num_heads=args.bottleneck_num_heads,
        dec_num_heads=args.dec_num_heads,
        downsample=args.downsample,
        upsample=args.upsample,
        drop_rate=args.dr,
        drop_path_rate=args.dpr,
        attn_drop_rate=args.adr,
        attn_out_drop_rate=args.aodr,
        pos_drop_rate=args.posdr,
        geodesic_bias_mode=args.geodesic_bias_mode,
        geodesic_bias_scale=args.geodesic_bias_scale,
        acuity_stream_layers=args.acuity_stream_layers,
        acuity_stream_heads=args.acuity_stream_heads,
        acuity_stream_dim_ratio=args.acuity_stream_dim_ratio,
        acuity_gate_init=args.acuity_gate_init,
        contextual_ambiguity_aware_geodesic_bias_s_min=(
            args.contextual_ambiguity_aware_geodesic_bias_s_min
        ),
        contextual_ambiguity_aware_geodesic_bias_s_max=(
            args.contextual_ambiguity_aware_geodesic_bias_s_max
        ),
    )


def _load_checkpoint_state(ckpt_path: str, map_location=None):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=map_location)
    state = state.get("state_dict", state.get("model", state))
    if not isinstance(state, dict):
        raise RuntimeError(f"[predict] Unsupported checkpoint format at {ckpt_path}")
    return {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, map_location=None, state=None):
    if state is None:
        state = _load_checkpoint_state(ckpt_path, map_location=map_location)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        warnings.warn(f"[predict] Missing keys: {list(missing)}")
    if unexpected:
        raise RuntimeError(f"[predict] Unexpected keys in checkpoint: {list(unexpected)}")


def make_erp_to_node_nn_map(width: int, normals_xyz: np.ndarray, chunk: int = 8192) -> np.ndarray:
    width = int(width)
    height = width // 2

    xs = np.linspace(-np.pi, np.pi, width, endpoint=False, dtype=np.float32)
    ys = np.linspace(np.pi / 2, -np.pi / 2, height, endpoint=True, dtype=np.float32)

    node_vecs = normals_xyz.astype(np.float32)
    idx_map = np.empty((height, width), dtype=np.int64)

    row_step = 64
    for y0 in range(0, height, row_step):
        y1 = min(height, y0 + row_step)
        phi = ys[y0:y1][:, None]
        theta = xs[None, :]
        sin_phi = np.broadcast_to(np.sin(phi), (y1 - y0, width))
        cos_phi = np.broadcast_to(np.cos(phi), (y1 - y0, width))

        xyz = np.stack(
            (
                cos_phi * np.cos(theta),
                sin_phi,
                cos_phi * np.sin(theta),
            ),
            axis=-1,
        ).reshape(-1, 3)

        out = np.empty((xyz.shape[0],), dtype=np.int64)
        for i in range(0, xyz.shape[0], chunk):
            j = min(xyz.shape[0], i + chunk)
            sims = xyz[i:j] @ node_vecs.T
            out[i:j] = np.argmax(sims, axis=1)

        idx_map[y0:y1] = out.reshape(y1 - y0, width)

    return idx_map


def colorize_labels(labels_hw: np.ndarray, colors: np.ndarray) -> np.ndarray:
    labels_hw = labels_hw.astype(np.int64, copy=False)
    labels_hw = np.clip(labels_hw, 0, colors.shape[0] - 1)
    return colors[labels_hw].astype(np.uint8)


def overlay_on_image(img_rgb: np.ndarray, seg_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    img = img_rgb.astype(np.float32)
    seg = seg_rgb.astype(np.float32)
    if seg.shape[:2] != img.shape[:2]:
        seg = cv2.resize(seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return (alpha * seg + (1 - alpha) * img).clip(0, 255).astype(np.uint8)


def apply_mask_as_black(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if img_rgb.shape[:2] != mask.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {img_rgb.shape[:2]}")

    out = img_rgb.copy()
    out[mask == 0] = 0
    return out


def crop_to_mask_rows(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask > 0, axis=1)
    top = int(np.argmax(rows))
    bottom = int(len(rows) - np.argmax(rows[::-1]))
    return img_rgb[top:bottom]


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a single ERP image with AdapToPASS")
    parser.add_argument("--image_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(CHECKPOINT_DIR, "model_best.pth"),
    )
    parser.add_argument(
        "--colors_path",
        type=str,
        default=os.path.join(DATA_DIR, "stanford2d3d_colors.npy"),
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=MASK_PATH,
    )
    parser.add_argument("--gt_path", type=str, default=None)

    parser.add_argument("--mode", type=str, default="vertex", choices=["face", "vertex"])
    parser.add_argument("--img_rank", type=int, default=7)
    parser.add_argument("--img_width", type=int, default=512)
    parser.add_argument("--num_scales", type=int, default=4)
    parser.add_argument("--win_size_coef", type=int, default=2)
    parser.add_argument("--scale_factor", type=int, default=2)
    parser.add_argument("--d_head_coef", type=int, default=2)
    parser.add_argument("--enc_num_heads", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--dec_num_heads", nargs="+", type=int, default=[16, 16, 8, 4])
    parser.add_argument("--bottleneck_num_heads", type=int, default=16)
    parser.add_argument("--scale_layers", type=int, default=2)

    parser.add_argument("--dr", type=float, default=0.0)
    parser.add_argument("--dpr", type=float, default=0.0)
    parser.add_argument("--adr", type=float, default=0.0)
    parser.add_argument("--aodr", type=float, default=0.0)
    parser.add_argument("--posdr", type=float, default=0.0)
    parser.add_argument("--geodesic_bias_mode", type=str, default="mlp", choices=["mlp", "rbf"])
    parser.add_argument("--geodesic_bias_scale", type=float, default=1.0)
    parser.add_argument("--acuity_stream_layers", type=int, default=1)
    parser.add_argument("--acuity_stream_heads", type=int, default=2)
    parser.add_argument("--acuity_stream_dim_ratio", type=int, default=2)
    parser.add_argument("--acuity_gate_init", type=float, default=-2.0)
    parser.add_argument("--contextual_ambiguity_aware_geodesic_bias_s_min", type=float, default=0.8)
    parser.add_argument("--contextual_ambiguity_aware_geodesic_bias_s_max", type=float, default=1.6)

    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--nn_chunk", type=int, default=8192)
    parser.add_argument("--downsample", type=str, default="center")
    parser.add_argument("--upsample", type=str, default="interpolate")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = ensure_dir(args.output_dir)
    _emit_banner(args.image_path, out_dir, device)
    stem = os.path.splitext(os.path.basename(args.image_path))[0]

    erp_rgb_full = read_rgb(args.image_path)

    width = args.img_width
    height = width // 2
    erp_rgb = cv2.resize(erp_rgb_full, (width, height), interpolation=cv2.INTER_AREA)

    colors = load_colors(args.colors_path)
    num_classes = colors.shape[0]

    with _suppress_internal_output():
        ckpt_state = _load_checkpoint_state(args.checkpoint, map_location=device)
        model = build_model(num_classes=num_classes, args=args).to(device).eval()
        load_checkpoint(model, args.checkpoint, map_location=device, state=ckpt_state)

    x_nodes, normals_wh = to_tensor_sphere_rgb_like_dataset(erp_rgb, args.img_rank, args.mode)
    x_nodes = normalize_like_training(x_nodes.to(device))

    with torch.no_grad():
        outputs = model(x_nodes)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs

    if logits.dim() != 3 or logits.size(0) != 1:
        raise RuntimeError(f"[predict] Expected logits shaped [1, N, C], got {tuple(logits.shape)}")
    logits_np = logits.squeeze(0).detach().cpu().numpy()
    node_labels = np.argmax(logits_np, axis=1).astype(np.int64)

    theta = normals_wh[:, 0] * 180.0
    phi = (normals_wh[:, 1] + 1.0) * 0.5 * 180.0
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    node_xyz = np.stack(
        (
            np.sin(phi_rad) * np.cos(theta_rad),
            np.cos(phi_rad),
            np.sin(phi_rad) * np.sin(theta_rad),
        ),
        axis=1,
    ).astype(np.float32)

    cache_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    idx_cache = os.path.join(cache_dir, f"erp2node_W{width}_rank{args.img_rank}_{args.mode}.npy")
    if os.path.isfile(idx_cache):
        idx_map = np.load(idx_cache)
        if idx_map.shape != (height, width):
            warnings.warn(f"[predict] Cached index map shape {idx_map.shape} != {(height, width)}. Recomputing.")
            idx_map = make_erp_to_node_nn_map(width, node_xyz, chunk=args.nn_chunk)
            np.save(idx_cache, idx_map)
    else:
        idx_map = make_erp_to_node_nn_map(width, node_xyz, chunk=args.nn_chunk)
        np.save(idx_cache, idx_map)

    erp_pred_labels = node_labels[idx_map]
    seg_rgb = colorize_labels(erp_pred_labels, colors)
    overlay = overlay_on_image(erp_rgb, seg_rgb, alpha=0.55)

    mask = load_binary_mask(args.mask_path, out_hw=seg_rgb.shape[:2])
    gt_path = args.gt_path or resolve_gt_path(args.image_path)
    if gt_path is not None:
        gt_mask = load_gt_ignore_mask(gt_path, out_hw=seg_rgb.shape[:2], ignore_rgb=colors[0])
        mask = np.minimum(mask, gt_mask)
    elif args.gt_path:
        warnings.warn(f"[predict] GT path was provided but not found: {args.gt_path}")
    seg_out = crop_to_mask_rows(apply_mask_as_black(seg_rgb, mask), mask)
    overlay_out = crop_to_mask_rows(apply_mask_as_black(overlay, mask), mask)

    pred_jpg = os.path.join(out_dir, f"{stem}_pred_labels.jpg")
    overlay_jpg = os.path.join(out_dir, f"{stem}_overlay.jpg")
    if not cv2.imwrite(pred_jpg, cv2.cvtColor(seg_out, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"[predict] Failed to save prediction image: {pred_jpg}")
    if not cv2.imwrite(overlay_jpg, cv2.cvtColor(overlay_out, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"[predict] Failed to save overlay image: {overlay_jpg}")

    dt = time.time() - t0
    _emit_success(pred_jpg, overlay_jpg, dt)
    _emit_ascii_art()


if __name__ == "__main__":
    main()
