from dataclasses import dataclass
from typing import Literal, Optional, Union

import cv2
import numpy as np
import torch
import tyro

from InfiniDepth.utils.inference_utils import (
    OUTPUT_RESOLUTION_MODES,
    apply_sky_mask_to_depth,
    build_scaled_intrinsics_matrix,
    resolve_camera_intrinsics_for_inference,
    resolve_depth_output_paths,
    resolve_output_size_from_mode,
    run_optional_sky_mask,
)
from InfiniDepth.utils.io_utils import (
    depth2pcd,
    load_image,
    plot_depth,
    save_depth_array,
    save_sampled_point_clouds,
)
from InfiniDepth.utils.model_utils import build_model
from InfiniDepth.utils.sampling_utils import SAMPLING_METHODS


@dataclass
class DepthInferenceArgs:
    input_image_path: str

    depth_output_dir: Optional[str] = None
    pcd_output_dir: Optional[str] = None
    save_pcd: bool = True

    model_type: str = "InfiniDepth"
    depth_model_path: str = "checkpoints/depth/infinidepth.ckpt"

    fx_org: Optional[float] = None
    fy_org: Optional[float] = None
    cx_org: Optional[float] = None
    cy_org: Optional[float] = None

    input_size: tuple[int, int] = (768, 1024)
    output_size: tuple[int, int] = (768, 1024)
    output_resolution_mode: Literal["upsample", "original", "specific"] = "upsample"
    upsample_ratio: int = 1

    enable_skyseg_model: bool = False
    sky_model_ckpt_path: str = "checkpoints/sky/skyseg.onnx"


@dataclass
class DepthInferenceResult:
    input_image_path: str
    org_img: torch.Tensor
    image: torch.Tensor
    query_2d_uniform_coord: torch.Tensor
    pred_2d_uniform_depth: torch.Tensor
    pred_depthmap: torch.Tensor
    org_h: int
    org_w: int
    input_h: int
    input_w: int
    output_h: int
    output_w: int
    fx_org: float
    fy_org: float
    cx_org: float
    cy_org: float
    fx: float
    fy: float
    cx: float
    cy: float
    intrinsics_source: str

    def output_intrinsics_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )


def load_depth_model(args: DepthInferenceArgs) -> tuple[torch.nn.Module, torch.device]:
    if args.output_resolution_mode not in OUTPUT_RESOLUTION_MODES:
        raise ValueError(
            f"Unsupported output_resolution_mode: {args.output_resolution_mode}. "
            f"Choose from {OUTPUT_RESOLUTION_MODES}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference in this script.")

    model = build_model(
        args.model_type,
        model_path=args.depth_model_path,
    )
    print(f"Loaded model: {model.__class__.__name__}")
    return model, torch.device("cuda")


def prepare_image_tensor(
    image_data: Union[torch.Tensor, np.ndarray],
    input_size: tuple[int, int] = (768, 1024),
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Convert an in-memory RGB image to the same format as ``load_image``.

    Args:
        image_data: RGB image, either (H, W, 3) uint8 [0,255] or float [0,1],
            or (3, H, W) / (1, 3, H, W) float tensors.
        input_size: Target (h, w) for the resized model input.

    Returns:
        org_img: (1, 3, org_h, org_w) float32 [0,1]
        image:   (1, 3, tar_h, tar_w) float32 [0,1]
        org_h, org_w: original spatial dimensions
    """
    if isinstance(image_data, torch.Tensor):
        t = image_data.detach().cpu().float()
        if t.ndim == 4 and t.shape[0] == 1:
            t = t.squeeze(0)
        if t.ndim == 3 and t.shape[0] in (1, 3):
            t = t.permute(1, 2, 0)
        if t.max() > 1.5:
            t = t / 255.0
        np_img = (t.clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    else:
        np_img = np.asarray(image_data)
        if np_img.ndim == 3 and np_img.shape[0] in (1, 3) and np_img.shape[2] not in (1, 3):
            np_img = np.moveaxis(np_img, 0, -1)
        if np_img.dtype != np.uint8:
            if np_img.max() <= 1.5:
                np_img = (np.clip(np_img, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                np_img = np.clip(np_img, 0, 255).astype(np.uint8)

    org_h, org_w = np_img.shape[:2]
    org_img = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    resized = cv2.resize(np_img, (input_size[1], input_size[0]), interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return org_img, image, org_h, org_w


@torch.no_grad()
def run_depth_inference_from_image(
    args: DepthInferenceArgs,
    image_data: Union[torch.Tensor, np.ndarray],
    *,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    fx_org: Optional[float] = None,
    fy_org: Optional[float] = None,
    cx_org: Optional[float] = None,
    cy_org: Optional[float] = None,
) -> "DepthInferenceResult":
    """Same as ``run_depth_inference`` but accepts an in-memory RGB image
    instead of a file path.  This avoids disk I/O and is the preferred entry
    point when images are already loaded (e.g. from ``Camera.toImageVis()``).
    """
    if model is None or device is None:
        model, device = load_depth_model(args)

    org_img, image, org_h, org_w = prepare_image_tensor(image_data, args.input_size)
    image = image.to(device)

    return _run_depth_inference_core(
        args=args,
        model=model,
        org_img=org_img,
        image=image,
        org_h=org_h,
        org_w=org_w,
        frame_image_path="<in-memory>",
        fx_org=fx_org,
        fy_org=fy_org,
        cx_org=cx_org,
        cy_org=cy_org,
    )


@torch.no_grad()
def _run_depth_inference_core(
    args: DepthInferenceArgs,
    model: torch.nn.Module,
    org_img: torch.Tensor,
    image: torch.Tensor,
    org_h: int,
    org_w: int,
    frame_image_path: str,
    fx_org: Optional[float],
    fy_org: Optional[float],
    cx_org: Optional[float],
    cy_org: Optional[float],
) -> "DepthInferenceResult":
    """Shared inference logic used by both file-path and in-memory entry points."""

    frame_fx_org = args.fx_org if fx_org is None else fx_org
    frame_fy_org = args.fy_org if fy_org is None else fy_org
    frame_cx_org = args.cx_org if cx_org is None else cx_org
    frame_cy_org = args.cy_org if cy_org is None else cy_org

    frame_fx_org, frame_fy_org, frame_cx_org, frame_cy_org, intrinsics_source = resolve_camera_intrinsics_for_inference(
        frame_fx_org,
        frame_fy_org,
        frame_cx_org,
        frame_cy_org,
        org_h,
        org_w,
    )
    if intrinsics_source == "default":
        print(
            "Camera intrinsics are partially/fully missing. "
            f"Using image-size defaults in original space: fx={frame_fx_org:.2f}, fy={frame_fy_org:.2f}, cx={frame_cx_org:.2f}, cy={frame_cy_org:.2f}"
        )

    _, _, h, w = image.shape
    fx, fy, cx, cy, _ = build_scaled_intrinsics_matrix(
        fx_org=frame_fx_org,
        fy_org=frame_fy_org,
        cx_org=frame_cx_org,
        cy_org=frame_cy_org,
        org_h=org_h,
        org_w=org_w,
        h=h,
        w=w,
        device=image.device,
    )
    print(f"Scaled Intrinsics: fx {fx:.2f}, fy {fy:.2f}, cx {cx:.2f}, cy {cy:.2f}")

    sky_mask = run_optional_sky_mask(
        image=image,
        enable_skyseg_model=args.enable_skyseg_model,
        sky_model_ckpt_path=args.sky_model_ckpt_path,
    )

    h_sample, w_sample = resolve_output_size_from_mode(
        output_resolution_mode=args.output_resolution_mode,
        org_h=org_h,
        org_w=org_w,
        h=h,
        w=w,
        output_size=args.output_size,
        upsample_ratio=args.upsample_ratio,
    )

    query_2d_uniform_coord = SAMPLING_METHODS["2d_uniform"]((h_sample, w_sample)).unsqueeze(0).to(image.device)
    pred_2d_uniform_depth, _ = model.inference(
        image=image,
        query_coord=query_2d_uniform_coord,
        gt_depth=None,
        gt_depth_mask=None,
        prompt_depth=None,
        prompt_mask=None,
    )
    pred_depthmap = pred_2d_uniform_depth.permute(0, 2, 1).view(1, 1, h_sample, w_sample)

    pred_depthmap, pred_2d_uniform_depth = apply_sky_mask_to_depth(
        pred_depthmap=pred_depthmap,
        pred_2d_uniform_depth=pred_2d_uniform_depth,
        sky_mask=sky_mask,
        h_sample=h_sample,
        w_sample=w_sample,
        sky_depth_value=200.0,
    )

    return DepthInferenceResult(
        input_image_path=frame_image_path,
        org_img=org_img,
        image=image,
        query_2d_uniform_coord=query_2d_uniform_coord,
        pred_2d_uniform_depth=pred_2d_uniform_depth,
        pred_depthmap=pred_depthmap,
        org_h=org_h,
        org_w=org_w,
        input_h=h,
        input_w=w,
        output_h=h_sample,
        output_w=w_sample,
        fx_org=frame_fx_org,
        fy_org=frame_fy_org,
        cx_org=frame_cx_org,
        cy_org=frame_cy_org,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        intrinsics_source=intrinsics_source,
    )


@torch.no_grad()
def run_depth_inference(
    args: DepthInferenceArgs,
    *,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    input_image_path: Optional[str] = None,
    fx_org: Optional[float] = None,
    fy_org: Optional[float] = None,
    cx_org: Optional[float] = None,
    cy_org: Optional[float] = None,
) -> DepthInferenceResult:
    if model is None or device is None:
        model, device = load_depth_model(args)

    frame_image_path = input_image_path or args.input_image_path

    org_img, image, (org_h, org_w) = load_image(frame_image_path, args.input_size)
    image = image.to(device)

    return _run_depth_inference_core(
        args=args,
        model=model,
        org_img=org_img,
        image=image,
        org_h=org_h,
        org_w=org_w,
        frame_image_path=frame_image_path,
        fx_org=fx_org,
        fy_org=fy_org,
        cx_org=cx_org,
        cy_org=cy_org,
    )


def build_point_cloud_from_depth_result(
    result: DepthInferenceResult,
    *,
    pcd_extrinsics_w2c: Optional[np.ndarray] = None,
    pcd_intrinsics_override: Optional[np.ndarray] = None,
    filter_flying_points: bool = True,
    nb_neighbors: int = 30,
    std_ratio: float = 2.0,
):
    pcd_intrinsics = result.output_intrinsics_matrix()
    if pcd_intrinsics_override is not None:
        pcd_intrinsics = np.asarray(pcd_intrinsics_override, dtype=np.float32)
        if pcd_intrinsics.shape != (3, 3):
            raise ValueError(
                f"pcd_intrinsics_override must have shape (3, 3), got {pcd_intrinsics.shape}"
            )
    pcd = depth2pcd(
        result.query_2d_uniform_coord.squeeze().cpu(),
        result.pred_2d_uniform_depth.squeeze().cpu(),
        result.image.squeeze().cpu(),
        pcd_intrinsics,
        ext=pcd_extrinsics_w2c,
    )
    if filter_flying_points and len(pcd.points) > 0:
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        pcd = pcd.select_by_index(ind)
    return pcd


def save_depth_inference_result(
    result: DepthInferenceResult,
    *,
    depth_vis_path: str,
    depth_raw_path: Optional[str] = None,
    pcd_path: Optional[str] = None,
    save_pcd: bool = True,
    pcd_extrinsics_w2c: Optional[np.ndarray] = None,
    pcd_intrinsics_override: Optional[np.ndarray] = None,
):
    plot_depth(result.org_img, result.pred_depthmap, depth_vis_path)
    if depth_raw_path is not None:
        save_depth_array(result.pred_depthmap, depth_raw_path)

    if not save_pcd:
        return None

    if pcd_path is not None:
        return save_sampled_point_clouds(
            result.query_2d_uniform_coord.squeeze().cpu(),
            result.pred_2d_uniform_depth.squeeze().cpu(),
            result.image.squeeze().cpu(),
            result.fx,
            result.fy,
            result.cx,
            result.cy,
            pcd_path,
            ixt=pcd_intrinsics_override,
            extrinsics_w2c=pcd_extrinsics_w2c,
        )

    return build_point_cloud_from_depth_result(
        result,
        pcd_extrinsics_w2c=pcd_extrinsics_w2c,
        pcd_intrinsics_override=pcd_intrinsics_override,
    )


@torch.no_grad()
def main(args: DepthInferenceArgs) -> None:
    model, device = load_depth_model(args)
    result = run_depth_inference(args, model=model, device=device)

    output_paths = resolve_depth_output_paths(
        input_image_path=args.input_image_path,
        model_type=args.model_type,
        output_resolution_mode=args.output_resolution_mode,
        upsample_ratio=args.upsample_ratio,
        h_sample=result.output_h,
        w_sample=result.output_w,
        depth_output_dir=args.depth_output_dir,
        pcd_output_dir=args.pcd_output_dir,
    )

    save_depth_inference_result(
        result,
        depth_vis_path=output_paths.depth_path,
        pcd_path=output_paths.pcd_path if args.save_pcd else None,
        save_pcd=args.save_pcd,
    )


if __name__ == "__main__":
    main(tyro.cli(DepthInferenceArgs))
