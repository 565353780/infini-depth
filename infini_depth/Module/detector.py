import os
import torch
import numpy as np

from copy import deepcopy
from typing import List, Optional, Union

from camera_control.Module.camera import Camera
from camera_control.Method.data import toNumpy

from inference_depth import (
    DepthInferenceArgs,
    DepthInferenceResult,
    load_depth_model,
    run_depth_inference_from_image,
)
from InfiniDepth.utils.inference_utils import ensure_homogeneous_extrinsics


class Detector(object):
    def __init__(
        self,
        model_file_path: Optional[str] = None,
        model_type: str = "InfiniDepth",
        input_size: tuple = (768, 1024),
        device: str = "cuda:0",
    ) -> None:
        self.model_type = model_type
        self.input_size = input_size
        self.device_str = device

        self.model: Optional[torch.nn.Module] = None
        self.torch_device: Optional[torch.device] = None
        self._default_args: Optional[DepthInferenceArgs] = None

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(
        self,
        model_file_path: str,
    ) -> bool:
        if not os.path.exists(model_file_path):
            print('[ERROR][Detector::loadModel]')
            print('\t model file not exist!')
            print('\t model_file_path:', model_file_path)
            return False

        self._default_args = DepthInferenceArgs(
            input_image_path="<placeholder>",
            model_type=self.model_type,
            depth_model_path=model_file_path,
            input_size=self.input_size,
            output_resolution_mode="original",
        )

        self.model, self.torch_device = load_depth_model(self._default_args)

        print('[INFO][Detector::loadModel]')
        print('\t model loaded from:', model_file_path)
        return True

    @property
    def is_valid(self) -> bool:
        return self.model is not None and self._default_args is not None

    @staticmethod
    def _normalize_images(
        images: Union[List[torch.Tensor], List[np.ndarray], torch.Tensor, np.ndarray],
    ) -> List[np.ndarray]:
        """Normalize heterogeneous image inputs to a list of (H, W, 3) uint8 numpy arrays."""
        if isinstance(images, torch.Tensor):
            if images.ndim == 4:
                images = [images[i] for i in range(images.shape[0])]
            elif images.ndim == 3:
                images = [images]
            else:
                raise ValueError(f"Unsupported torch.Tensor image shape: {images.shape}")
        elif isinstance(images, np.ndarray):
            if images.ndim == 4:
                images = [images[i] for i in range(images.shape[0])]
            elif images.ndim == 3:
                images = [images]
            else:
                raise ValueError(f"Unsupported np.ndarray image shape: {images.shape}")

        result = []
        for img in images:
            if isinstance(img, torch.Tensor):
                t = img.detach().cpu().float()
                if t.ndim == 3 and t.shape[0] in (1, 3):
                    t = t.permute(1, 2, 0)
                if t.max() > 1.5:
                    t = t / 255.0
                arr = (t.clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            else:
                arr = np.asarray(img)
                if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
                    arr = np.moveaxis(arr, 0, -1)
                if arr.dtype != np.uint8:
                    if arr.max() <= 1.5:
                        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                    else:
                        arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"Expected (H, W, 3) image, got shape {arr.shape}")
            result.append(arr)
        return result

    @staticmethod
    def _normalize_intrinsics(
        intrinsics: Optional[Union[List, torch.Tensor, np.ndarray]],
        num_views: int,
    ) -> Optional[np.ndarray]:
        """Normalize to (N, 3, 3) float32 numpy, or None."""
        if intrinsics is None:
            return None
        if isinstance(intrinsics, torch.Tensor):
            intrinsics = intrinsics.detach().cpu().numpy()
        elif isinstance(intrinsics, list):
            intrinsics = np.array([
                k.detach().cpu().numpy() if isinstance(k, torch.Tensor) else np.asarray(k)
                for k in intrinsics
            ])
        intrinsics = np.asarray(intrinsics, dtype=np.float32)
        if intrinsics.ndim == 2:
            intrinsics = intrinsics[None, ...]
        if intrinsics.shape[0] == 1 and num_views > 1:
            intrinsics = np.tile(intrinsics, (num_views, 1, 1))
        if intrinsics.shape != (num_views, 3, 3):
            raise ValueError(
                f"Expected intrinsics shape ({num_views}, 3, 3), got {intrinsics.shape}"
            )
        return intrinsics

    @staticmethod
    def _normalize_extrinsics(
        extrinsics: Optional[Union[List, torch.Tensor, np.ndarray]],
        num_views: int,
    ) -> Optional[np.ndarray]:
        """Normalize to (N, 4, 4) float32 numpy, or None."""
        if extrinsics is None:
            return None
        if isinstance(extrinsics, torch.Tensor):
            extrinsics = extrinsics.detach().cpu().numpy()
        elif isinstance(extrinsics, list):
            extrinsics = np.array([
                e.detach().cpu().numpy() if isinstance(e, torch.Tensor) else np.asarray(e)
                for e in extrinsics
            ])
        extrinsics = np.asarray(extrinsics, dtype=np.float32)
        if extrinsics.ndim == 2:
            extrinsics = extrinsics[None, ...]
        extrinsics = ensure_homogeneous_extrinsics(extrinsics)
        if extrinsics.shape[0] == 1 and num_views > 1:
            extrinsics = np.tile(extrinsics, (num_views, 1, 1))
        if extrinsics.shape != (num_views, 4, 4):
            raise ValueError(
                f"Expected extrinsics shape ({num_views}, 4, 4), got {extrinsics.shape}"
            )
        return extrinsics

    @torch.no_grad()
    def detectImages(
        self,
        images: Union[List[torch.Tensor], List[np.ndarray], torch.Tensor, np.ndarray],
        intrinsics: Optional[Union[List, torch.Tensor, np.ndarray]] = None,
        extrinsics: Optional[Union[List, torch.Tensor, np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """Run depth inference on multiple views.

        Args:
            images: Multi-view RGB images in any common format.
            intrinsics: Optional (N, 3, 3) camera intrinsics.
            extrinsics: Optional (N, 4, 4) world-to-camera extrinsics (kept for
                API consistency; not used by the per-frame depth model itself).

        Returns:
            List of depth maps, each as a (H_out, W_out) float32 numpy array.
        """
        assert self.is_valid, (
            "Model not loaded. Call loadModel() first."
        )

        image_list = self._normalize_images(images)
        num_views = len(image_list)
        int_arr = self._normalize_intrinsics(intrinsics, num_views)
        self._normalize_extrinsics(extrinsics, num_views)

        depth_maps: List[np.ndarray] = []

        for i, img_np in enumerate(image_list):
            fx_org = float(int_arr[i, 0, 0]) if int_arr is not None else None
            fy_org = float(int_arr[i, 1, 1]) if int_arr is not None else None
            cx_org = float(int_arr[i, 0, 2]) if int_arr is not None else None
            cy_org = float(int_arr[i, 1, 2]) if int_arr is not None else None

            result: DepthInferenceResult = run_depth_inference_from_image(
                self._default_args,
                img_np,
                model=self.model,
                device=self.torch_device,
                fx_org=fx_org,
                fy_org=fy_org,
                cx_org=cx_org,
                cy_org=cy_org,
            )

            depth_np = result.pred_depthmap.squeeze().cpu().numpy().astype(np.float32)
            depth_maps.append(depth_np)

        return depth_maps

    @torch.no_grad()
    def detectCameras(
        self,
        camera_list: List[Camera],
    ) -> List[Camera]:
        """Collect images and camera parameters from Camera objects, run
        multi-view depth inference, and return new Camera objects with depth
        written back.

        Args:
            camera_list: Input cameras with images loaded.

        Returns:
            New list of Camera objects (deep-copied) with predicted depth.
        """
        if not camera_list:
            return []

        images: List[np.ndarray] = []
        intrinsics_list: List[np.ndarray] = []
        extrinsics_list: List[np.ndarray] = []

        for camera in camera_list:
            img_vis = camera.toImageVis(use_mask=False)
            img_np = (img_vis.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            images.append(img_np)

            intrinsics_list.append(toNumpy(camera.intrinsic, np.float32))
            extrinsics_list.append(toNumpy(camera.world2cameraCV, np.float32))

        intrinsics = np.stack(intrinsics_list, axis=0)
        extrinsics = np.stack(extrinsics_list, axis=0)

        depth_maps = self.detectImages(images, intrinsics=intrinsics, extrinsics=extrinsics)

        result_cameras = deepcopy(camera_list)
        for i, camera in enumerate(result_cameras):
            camera.loadDepth(depth_maps[i])

        return result_cameras
