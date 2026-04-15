import os
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from .registry import register_model
from .block.config import dinov3_model_configs
from .block.implicit_decoder import ImplicitHead
from .block.convolution import BasicEncoder

acc_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else (torch.float16 if torch.cuda.is_available() else torch.float32)
)


def _resolve_local_dinov3_repo() -> str:
    """Always use the in-repo local DINOv3 torchhub path."""
    dinov3_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "block", "torchhub", "dinov3"))
    if not os.path.isdir(dinov3_repo):
        raise FileNotFoundError(
            "DINOv3 local torchhub repo not found at fixed path: "
            f"{dinov3_repo}"
        )
    return dinov3_repo


@dataclass
class _InferenceState:
    gt_depth: Optional[torch.Tensor] = None
    gt_depth_mask: Optional[torch.Tensor] = None
    prompt_depth: Optional[torch.Tensor] = None
    prompt_mask: Optional[torch.Tensor] = None
    reference_meta: Optional[torch.Tensor] = None
    query_coord: Optional[torch.Tensor] = None


class _BaseInfiniDepthModel(nn.Module):
    def __init__(
        self,
        model_path: Optional[str] = None,
        encoder: str = "vitl16",
    ):
        super().__init__()
        self.model_config = dinov3_model_configs[encoder]
        local_dinov3_repo = _resolve_local_dinov3_repo()
        self.pretrained = torch.hub.load(
            local_dinov3_repo,
            f"dinov3_{encoder}",
            source="local",
            pretrained=False,
        )
        self.patch_size = 16
        dim = self.pretrained.blocks[0].attn.qkv.in_features

        self.basic_encoder = BasicEncoder(
            input_dim=3,
            output_dim=128,
            stride=4,
        )
        self.depth_implicit_head = ImplicitHead(
            hidden_dim=dim,
            basic_dim=128,
            fusion_type="concat",
            out_dim=1,
            hidden_list=[1024, 256, 32],
        )
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self._init_variant_modules()

        if model_path is not None:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location="cpu")
                self.load_state_dict({k[9:]: v for k, v in checkpoint["state_dict"].items()})
            else:
                raise FileNotFoundError(f"Model file {model_path} not found")

        # only for inference
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to initialize InfiniDepth models.")
        self.cuda()
        self.eval()

    def _init_variant_modules(self):
        """Subclasses can attach extra modules or state."""

    def _transform_features(
        self,
        features,
        patch_h: int,
        patch_w: int,
        state: _InferenceState,
    ):
        """Subclasses can inject prompt/depth conditioning."""
        return features

    def _prepare_inference(self, state: _InferenceState) -> _InferenceState:
        """Subclasses can pre-process inference inputs before forward passes."""
        return state

    def _postprocess_inference(
        self,
        pred: torch.Tensor,
        image: torch.Tensor,
        state: _InferenceState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _prepare_backbone_features(
        self,
        x: torch.Tensor,
        state: _InferenceState,
    ):
        h, w = x.shape[-2:]
        x_dino = (x - self._mean) / self._std
        with torch.autocast("cuda", enabled=True, dtype=acc_dtype):
            features = self.pretrained.get_intermediate_layers(
                x_dino,
                n=self.model_config["layer_idxs"],
                return_class_token=True,
            )
        dino_tokens = features[-1][0].clone()
        features = [list(feature) for feature in features]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        features = self._transform_features(
            features,
            patch_h,
            patch_w,
            state,
        )

        x_basic = 2.0 * x - 1.0
        basic_feat = self.basic_encoder(x_basic)  # [B, 128, H/4, W/4]
        return features, basic_feat, patch_h, patch_w, dino_tokens

    def _to_depth_disparity(self, pred: torch.Tensor):
        pred_disparity = pred
        pred_depth = 1.0 / torch.clamp(pred, min=5e-3)
        return pred_depth, pred_disparity

    @torch.no_grad()
    def inference(
        self,
        image: torch.Tensor,
        query_coord: torch.Tensor,
        use_batch_infer=True,
        return_dino_tokens: bool = False,
        gt_depth: Optional[torch.Tensor] = None,
        gt_depth_mask: Optional[torch.Tensor] = None,
        prompt_depth: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
    ):
        state = _InferenceState(
            gt_depth=gt_depth,
            gt_depth_mask=gt_depth_mask,
            prompt_depth=prompt_depth,
            prompt_mask=prompt_mask,
            query_coord=query_coord,
        )
        state = self._prepare_inference(state)

        if use_batch_infer:
            pred = self.batch_forward(
                image,
                query_coord,
                bsize=10000,
                return_dino_tokens=return_dino_tokens,
                prompt_depth=state.prompt_depth,
                prompt_mask=state.prompt_mask,
            )
        else:
            pred = self.forward(
                image,
                query_coord,
                return_dino_tokens=return_dino_tokens,
                prompt_depth=state.prompt_depth,
                prompt_mask=state.prompt_mask,
            )

        if return_dino_tokens:
            pred, dino_tokens = pred

        pred_depth, pred_disparity = self._postprocess_inference(
            pred,
            image,
            state,
        )

        if return_dino_tokens:
            return pred_depth, pred_disparity, dino_tokens
        return pred_depth, pred_disparity

    def batch_forward(
        self,
        x: torch.Tensor,
        coord: torch.Tensor,
        prompt_depth: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        bsize: int = 3000,
        return_dino_tokens: bool = False,
    ):
        """Forward pass with batching to avoid OOM."""
        state = _InferenceState(prompt_depth=prompt_depth, prompt_mask=prompt_mask)
        features, basic_feat, patch_h, patch_w, dino_tokens = self._prepare_backbone_features(
            x,
            state=state,
        )
        feat = self.depth_implicit_head._encode_feat(features, patch_h, patch_w)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = self.depth_implicit_head._decode_dpt(feat, basic_feat, coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
        if return_dino_tokens:
            return pred, dino_tokens
        return pred

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        prompt_depth: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        return_dino_tokens: bool = False,
    ):
        state = _InferenceState(prompt_depth=prompt_depth, prompt_mask=prompt_mask)
        features, basic_feat, patch_h, patch_w, dino_tokens = self._prepare_backbone_features(
            x,
            state=state,
        )
        with torch.autocast("cuda", enabled=True, dtype=torch.float32):
            depth = self.depth_implicit_head(features, basic_feat, patch_h, patch_w, coords)
        if return_dino_tokens:
            return depth, dino_tokens
        return depth

@register_model("InfiniDepth")
class InfiniDepth(_BaseInfiniDepthModel):
    def _init_variant_modules(self):
        poly_features = PolynomialFeatures(degree=1, include_bias=False)
        ransac = RANSACRegressor(max_trials=1000)
        self.ransac_model = make_pipeline(poly_features, ransac)
        self._cached_denorm_scale: Optional[torch.Tensor] = None
        self._cached_denorm_shift: Optional[torch.Tensor] = None

    def _get_cached_denorm_params(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if self._cached_denorm_scale is None or self._cached_denorm_shift is None:
            return None

        scale = self._cached_denorm_scale
        shift = self._cached_denorm_shift
        if scale.numel() == 1 and batch > 1:
            scale = scale.expand(batch)
            shift = shift.expand(batch)
        elif batch == 1 and scale.numel() > 1:
            scale = scale[:1]
            shift = shift[:1]
        elif scale.numel() != batch:
            return None

        return (
            scale.to(device=device, dtype=dtype).reshape(batch, 1, 1),
            shift.to(device=device, dtype=dtype).reshape(batch, 1, 1),
        )

    def _ransac_align_depth(self, pred, gt, mask0=None):
        if type(pred).__module__ == torch.__name__:
            pred = pred.cpu().numpy()
        if type(gt).__module__ == torch.__name__:
            gt = gt.cpu().numpy()
        pred = pred.astype(np.float32)
        gt = gt.astype(np.float32)
        gt = gt.squeeze()
        pred = pred.squeeze()
        mask = (gt > 1e-8)  # & (pred > 1e-8)
        if mask0 is not None and mask0.sum() > 0:
            if type(mask0).__module__ == torch.__name__:
                mask0 = mask0.cpu().numpy()
            mask0 = mask0.squeeze()
            mask0 = mask0 > 0
            mask = mask & mask0
        gt_mask = gt[mask].astype(np.float32)
        pred_mask = pred[mask].astype(np.float32)

        gt_mask = np.clip(gt_mask, 1e-8, None)

        try:
            self.ransac_model.fit(pred_mask[:, None], gt_mask[:, None])
            a, b = (
                self.ransac_model.named_steps["ransacregressor"].estimator_.coef_,
                self.ransac_model.named_steps["ransacregressor"].estimator_.intercept_,
            )
            a = a.item()
            b = b.item()
        except Exception:
            a, b = 1, 0

        if not np.isfinite(a):
            a = 1.0
        if not np.isfinite(b):
            b = 0.0

        if a > 0:
            pred_metric = a * pred + b
        else:
            if pred_mask.size > 0 and gt_mask.size > 0:
                pred_mean = max(float(np.mean(pred_mask)), 1e-8)
                gt_mean = float(np.mean(gt_mask))
                a = gt_mean / pred_mean
            else:
                a = 1.0
            b = 0.0
            pred_metric = a * pred + b

        return torch.from_numpy(pred_metric).unsqueeze(0).unsqueeze(0), float(a), float(b)

    @staticmethod
    def _infer_dense_query_hw(query_coord: Optional[torch.Tensor], n_query: int) -> Optional[tuple[int, int]]:
        if query_coord is None or query_coord.ndim != 3 or query_coord.shape[1] != n_query:
            return None

        query0 = query_coord[0]
        if query0.ndim != 2 or query0.shape[1] != 2:
            return None

        h_query = int(torch.unique(query0[:, 0]).numel())
        w_query = int(torch.unique(query0[:, 1]).numel())
        if h_query * w_query != n_query:
            return None
        return h_query, w_query

    def _postprocess_inference(
        self,
        pred: torch.Tensor,
        image: torch.Tensor,
        state: _InferenceState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, _, _ = image.shape
        n_query = pred.shape[1]
        dense_query_hw = self._infer_dense_query_hw(state.query_coord, n_query)
        if dense_query_hw is not None and state.gt_depth is not None:
            h_query, w_query = dense_query_hw
            pred_map = pred.permute(0, 2, 1).reshape(b, 1, h_query, w_query)
            gt_align = state.gt_depth
            gt_mask_align = state.gt_depth_mask
            if gt_align.shape[-2:] != (h_query, w_query):
                gt_align = F.interpolate(
                    gt_align,
                    size=(h_query, w_query),
                    mode="bilinear",
                    align_corners=False,
                )
                if gt_mask_align is not None:
                    gt_mask_align = F.interpolate(
                        gt_mask_align.float(),
                        size=(h_query, w_query),
                        mode="nearest",
                    )
            aligned = []
            scales = []
            shifts = []
            for i in range(b):
                aligned_i, scale_i, shift_i = self._ransac_align_depth(
                    pred_map[i: i + 1],
                    gt_align[i: i + 1],
                    None if gt_mask_align is None else gt_mask_align[i: i + 1],
                )
                aligned_i = aligned_i.to(device=image.device, dtype=pred.dtype)
                aligned.append(aligned_i)
                scales.append(scale_i)
                shifts.append(shift_i)
            pred_map = torch.cat(aligned, dim=0)
            pred = pred_map.reshape(b, 1, -1).permute(0, 2, 1)
            self._cached_denorm_scale = torch.tensor(scales, dtype=torch.float32)
            self._cached_denorm_shift = torch.tensor(shifts, dtype=torch.float32)
        else:
            cached_params = self._get_cached_denorm_params(
                batch=b,
                device=pred.device,
                dtype=pred.dtype,
            )
            if cached_params is not None:
                scale, shift = cached_params
                pred = pred * scale + shift
        return self._to_depth_disparity(pred)
