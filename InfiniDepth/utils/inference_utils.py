import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .vis_utils import build_sky_model, run_skyseg


OUTPUT_RESOLUTION_MODES = ("upsample", "original", "specific")


@dataclass
class DepthOutputPaths:
    depth_output_dir: str
    pcd_output_dir: str
    depth_path: str
    pcd_path: str


def scale_intrinsics(fx, fy, cx, cy, org_h, org_w, h, w):
    sx, sy = w / float(org_w), h / float(org_h)
    return fx * sx, fy * sy, cx * sx, cy * sy


def resolve_camera_intrinsics(
    fx_org: Optional[float],
    fy_org: Optional[float],
    cx_org: Optional[float],
    cy_org: Optional[float],
    org_h: int,
    org_w: int,
    fallback_intrinsics: Optional[tuple[float, float, float, float]] = None,
) -> tuple[float, float, float, float]:
    default_focal = float(max(org_h, org_w))
    default_cx = float(org_w) / 2.0
    default_cy = float(org_h) / 2.0

    fallback_fx, fallback_fy, fallback_cx, fallback_cy = fallback_intrinsics or (
        default_focal,
        default_focal,
        default_cx,
        default_cy,
    )

    fx = float(fx_org) if fx_org is not None else float(fallback_fx)
    fy = float(fy_org) if fy_org is not None else float(fallback_fy)
    cx = float(cx_org) if cx_org is not None else float(fallback_cx)
    cy = float(cy_org) if cy_org is not None else float(fallback_cy)
    return fx, fy, cx, cy


def resolve_output_size_from_mode(
    output_resolution_mode: str,
    org_h: int,
    org_w: int,
    h: int,
    w: int,
    output_size: tuple[int, int],
    upsample_ratio: int,
) -> tuple[int, int]:
    if output_resolution_mode not in OUTPUT_RESOLUTION_MODES:
        raise ValueError(
            f"Unsupported output_resolution_mode: {output_resolution_mode}. "
            f"Choose from {OUTPUT_RESOLUTION_MODES}."
        )

    if output_resolution_mode == "specific":
        h_out, w_out = int(output_size[0]), int(output_size[1])
    elif output_resolution_mode == "original":
        h_out, w_out = int(org_h), int(org_w)
    else:
        if upsample_ratio < 1:
            raise ValueError("`upsample_ratio` must be >= 1 when output_resolution_mode=upsample.")
        h_out, w_out = int(h * upsample_ratio), int(w * upsample_ratio)

    if h_out <= 0 or w_out <= 0:
        raise ValueError(f"Invalid output size ({h_out}, {w_out}). Height and width must be positive.")
    return h_out, w_out


def default_dir_by_input_file(input_path: str, output_name: str) -> str:
    base_dir = Path(input_path).resolve().parent.parent
    return os.path.join(base_dir, output_name)


def resolve_camera_intrinsics_for_inference(
    fx_org: Optional[float],
    fy_org: Optional[float],
    cx_org: Optional[float],
    cy_org: Optional[float],
    org_h: int,
    org_w: int,
) -> tuple[float, float, float, float, str]:
    intrinsics_source = "user"
    if any(v is None for v in (fx_org, fy_org, cx_org, cy_org)):
        intrinsics_source = "default"
    fx, fy, cx, cy = resolve_camera_intrinsics(
        fx_org=fx_org, fy_org=fy_org, cx_org=cx_org, cy_org=cy_org,
        org_h=org_h, org_w=org_w,
    )
    return fx, fy, cx, cy, intrinsics_source


def build_scaled_intrinsics_matrix(
    fx_org: float,
    fy_org: float,
    cx_org: float,
    cy_org: float,
    org_h: int,
    org_w: int,
    h: int,
    w: int,
    device: torch.device,
) -> tuple[float, float, float, float, torch.Tensor]:
    fx, fy, cx, cy = scale_intrinsics(fx_org, fy_org, cx_org, cy_org, org_h, org_w, h, w)
    k = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    return fx, fy, cx, cy, k


def run_optional_sky_mask(
    image: torch.Tensor,
    enable_skyseg_model: bool,
    sky_model_ckpt_path: str,
) -> Optional[torch.Tensor]:
    if not enable_skyseg_model:
        return None

    if not os.path.exists(sky_model_ckpt_path):
        raise FileNotFoundError(
            f"Sky segmentation checkpoint not found: {sky_model_ckpt_path}. "
            "Disable `enable_skyseg_model` or provide a valid path."
        )

    _, _, h, w = image.shape
    sky_model = build_sky_model(model_path=sky_model_ckpt_path)
    sky_mask_np = run_skyseg(sky_model, input_size=(320, 320), image=image)
    sky_mask_np = cv2.resize(sky_mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(sky_mask_np).to(image.device)


def apply_sky_mask_to_depth(
    pred_depthmap: torch.Tensor,
    pred_2d_uniform_depth: torch.Tensor,
    sky_mask: Optional[torch.Tensor],
    h_sample: int,
    w_sample: int,
    sky_depth_value: float = 200.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sky_mask is None:
        return pred_depthmap, pred_2d_uniform_depth

    sky_mask_resized = (
        F.interpolate(
            sky_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(h_sample, w_sample),
            mode="nearest",
        )
        .bool()
        .squeeze()
    )
    pred_depthmap[:, :, sky_mask_resized] = sky_depth_value
    sky_mask_flat = sky_mask_resized.view(-1)
    pred_2d_uniform_depth[:, sky_mask_flat, :] = sky_depth_value
    return pred_depthmap, pred_2d_uniform_depth


def resolve_depth_output_paths(
    input_image_path: str,
    model_type: str,
    output_resolution_mode: str,
    upsample_ratio: int,
    h_sample: int,
    w_sample: int,
    depth_output_dir: Optional[str] = None,
    pcd_output_dir: Optional[str] = None,
) -> DepthOutputPaths:
    depth_dir = depth_output_dir or default_dir_by_input_file(input_image_path, "pred_depth")
    pcd_dir = pcd_output_dir or default_dir_by_input_file(input_image_path, "pred_pcd")

    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(pcd_dir, exist_ok=True)

    stem = os.path.basename(input_image_path).split(".")[0]
    if output_resolution_mode == "specific":
        depth_path = os.path.join(
            depth_dir,
            f"{model_type}_{stem}_{h_sample}x{w_sample}.png",
        )
        pcd_path = os.path.join(
            pcd_dir,
            f"{model_type}_{stem}_{h_sample}x{w_sample}.ply",
        )
    elif output_resolution_mode == "original":
        depth_path = os.path.join(depth_dir, f"{model_type}_{stem}_org_res.png")
        pcd_path = os.path.join(pcd_dir, f"{model_type}_{stem}_org_res.ply")
    else:
        depth_path = os.path.join(depth_dir, f"{model_type}_{stem}_up_{upsample_ratio}.png")
        pcd_path = os.path.join(pcd_dir, f"{model_type}_{stem}_up_{upsample_ratio}.ply")

    return DepthOutputPaths(
        depth_output_dir=depth_dir,
        pcd_output_dir=pcd_dir,
        depth_path=depth_path,
        pcd_path=pcd_path,
    )


def ensure_homogeneous_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics_np = np.asarray(extrinsics, dtype=np.float32)
    if extrinsics_np.ndim == 2:
        extrinsics_np = extrinsics_np[None, ...]

    if extrinsics_np.ndim != 3:
        raise ValueError(f'extrinsics must have 2 or 3 dims, got {extrinsics_np.shape}')

    if extrinsics_np.shape[-2:] == (4, 4):
        return extrinsics_np
    if extrinsics_np.shape[-2:] != (3, 4):
        raise ValueError(f'extrinsics must have shape (...,3,4) or (...,4,4), got {extrinsics_np.shape}')

    batch = extrinsics_np.shape[0]
    homogeneous = np.tile(np.eye(4, dtype=np.float32)[None, ...], (batch, 1, 1))
    homogeneous[:, :3, :] = extrinsics_np
    return homogeneous


def scale_intrinsics_matrix_np(
    intrinsics: np.ndarray,
    src_h: int,
    src_w: int,
    dst_h: int,
    dst_w: int,
) -> np.ndarray:
    if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
        raise ValueError(
            f'Invalid source/destination size: src=({src_h}, {src_w}), dst=({dst_h}, {dst_w})'
        )

    intrinsics_np = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics_np.shape != (3, 3):
        raise ValueError(f'intrinsics must have shape (3,3), got {intrinsics_np.shape}')

    fx, fy, cx, cy = scale_intrinsics(
        fx=float(intrinsics_np[0, 0]),
        fy=float(intrinsics_np[1, 1]),
        cx=float(intrinsics_np[0, 2]),
        cy=float(intrinsics_np[1, 2]),
        org_h=src_h,
        org_w=src_w,
        h=dst_h,
        w=dst_w,
    )
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
