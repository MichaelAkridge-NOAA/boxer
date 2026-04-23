# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
SAM3 text-prompt detector wrapper.

SAM3 (Segment Anything Model 3) supports open-vocabulary text prompts such as
"coral", "algae", "rubble" to produce pixel-level segmentation masks without
requiring a separate grounding model.

Both SAM3 and Boxer weights are available from:
    https://huggingface.co/facebook/sam3

Usage
-----
    from sam3.sam3_detector import SAM3Detector

    detector = SAM3Detector(ckpt_dir="./ckpts", device="cuda")
    bb2d, masks, scores = detector.detect(img_np, text_prompts=["coral"])

    # Video mode – temporally consistent instance IDs
    results = detector.detect_video(frames, text_prompts=["coral", "algae"])
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# HuggingFace repo ID for SAM3 + Boxer weights
# ---------------------------------------------------------------------------
_HF_REPO_ID = "facebook/sam3"

# Default checkpoint filename for the SAM3 model inside the HF repo
_SAM3_CKPT_NAME = "sam3.pt"


def _auto_download(ckpt_dir: str) -> str:
    """Download SAM3 weights from HuggingFace if not already cached.

    Returns the local directory containing the downloaded weights.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    sam3_ckpt = os.path.join(ckpt_dir, _SAM3_CKPT_NAME)
    if os.path.exists(sam3_ckpt):
        return ckpt_dir

    print(f"[SAM3] Downloading weights from HuggingFace ({_HF_REPO_ID}) …")
    try:
        from huggingface_hub import snapshot_download  # type: ignore

        local_dir = snapshot_download(
            repo_id=_HF_REPO_ID,
            local_dir=ckpt_dir,
            ignore_patterns=["*.md", "*.txt"],
        )
        print(f"[SAM3] Weights downloaded to {local_dir}")
        return local_dir
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download SAM3 weights from HuggingFace ({_HF_REPO_ID}). "
            "Please download manually and pass --sam3_ckpt pointing to the local folder.\n"
            f"Original error: {exc}"
        ) from exc


def _load_sam3_model(ckpt_dir: str, device: str):
    """Load the SAM3 model from a local checkpoint directory.

    Tries two backends in order:
      1. Ultralytics SAM3 (``ultralytics.models.sam.SAM`` with SAM3 weights)
      2. Native facebook/sam3 Python API (``sam3`` package)

    Returns a thin wrapper with a unified ``predict(image, text)`` interface.
    """
    # Locate checkpoint file
    sam3_ckpt = os.path.join(ckpt_dir, _SAM3_CKPT_NAME)
    if not os.path.exists(sam3_ckpt):
        # Some repos store it under a different name
        for candidate in os.listdir(ckpt_dir):
            if candidate.endswith(".pt") and "sam3" in candidate.lower():
                sam3_ckpt = os.path.join(ckpt_dir, candidate)
                break

    ul_error: Exception | None = None

    # --- Try Ultralytics SAM3 ---
    try:
        from ultralytics import SAM  # type: ignore

        model = SAM(sam3_ckpt)
        model.to(device)
        return _UltralyticsAdapter(model, device)
    except Exception as exc_ul:
        ul_error = exc_ul
        print(f"[SAM3] Ultralytics backend failed ({exc_ul}); trying native API …")

    # --- Try native sam3 package ---
    try:
        import sam3 as _sam3_pkg  # type: ignore

        model = _sam3_pkg.build_sam3(checkpoint=sam3_ckpt, device=device)
        predictor = _sam3_pkg.SAM3ImagePredictor(model)
        return _NativeSAM3Adapter(predictor, device)
    except Exception as exc_nat:
        raise RuntimeError(
            "Could not load SAM3 with either Ultralytics or the native sam3 package.\n"
            f"  Ultralytics error: {ul_error}\n"
            f"  Native error: {exc_nat}\n"
            "Install one of: `pip install ultralytics` or `pip install sam3`"
        ) from exc_nat


# ---------------------------------------------------------------------------
# Backend adapters
# ---------------------------------------------------------------------------


class _UltralyticsAdapter:
    """Adapts Ultralytics SAM3 to the SAM3Detector interface."""

    def __init__(self, model, device: str):
        self.model = model
        self.device = device

    def predict_image(
        self, img_np: np.ndarray, text_prompts: List[str]
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """Run text-prompted segmentation on a single RGB image (HxWx3 uint8).

        Returns:
            bb2d:   (M, 4) float32 [x0, y0, x1, y1]
            masks:  List of M (H, W) bool masks
            scores: (M,) float32 confidence scores
        """
        results = self.model.predict(
            source=img_np,
            texts=text_prompts,
            device=self.device,
            verbose=False,
        )
        return _parse_ultralytics_results(results, img_np.shape[:2])

    def predict_video(
        self, frames: List[np.ndarray], text_prompts: List[str]
    ) -> List[Tuple[np.ndarray, List[np.ndarray], np.ndarray]]:
        """Run text-prompted segmentation + tracking on a list of RGB frames."""
        results = self.model.track(
            source=frames,
            texts=text_prompts,
            device=self.device,
            verbose=False,
            stream=True,
        )
        out = []
        for res, frame in zip(results, frames):
            out.append(_parse_ultralytics_results([res], frame.shape[:2]))
        return out


class _NativeSAM3Adapter:
    """Adapts the native facebook/sam3 Python API to the SAM3Detector interface."""

    def __init__(self, predictor, device: str):
        self.predictor = predictor
        self.device = device

    def predict_image(
        self, img_np: np.ndarray, text_prompts: List[str]
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        self.predictor.set_image(img_np)
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            text_prompts=text_prompts,
            multimask_output=False,
        )
        bb2d = _masks_to_bboxes(masks)
        masks_list = [masks[i] for i in range(len(masks))]
        scores_arr = np.array(scores, dtype=np.float32).flatten()
        return bb2d, masks_list, scores_arr

    def predict_video(
        self, frames: List[np.ndarray], text_prompts: List[str]
    ) -> List[Tuple[np.ndarray, List[np.ndarray], np.ndarray]]:
        # Native API: process frame by frame (no built-in video tracking)
        return [self.predict_image(f, text_prompts) for f in frames]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_ultralytics_results(
    results, img_shape: Tuple[int, int]
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Extract bb2d / masks / scores from Ultralytics Results objects."""
    H, W = img_shape
    bb2d_list: List[np.ndarray] = []
    masks_list: List[np.ndarray] = []
    scores_list: List[float] = []

    for res in results:
        if res.masks is None:
            continue
        masks_tensor = res.masks.data  # (M, H', W') or (M, H, W)
        boxes = res.boxes
        for i in range(len(masks_tensor)):
            mask_hw = masks_tensor[i].cpu().numpy().astype(bool)
            # Resize mask to original resolution if needed
            if mask_hw.shape != (H, W):
                import cv2  # local import to avoid module-level dep at import time

                mask_hw = (
                    cv2.resize(
                        mask_hw.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                )
            masks_list.append(mask_hw)

            # Bounding box
            if boxes is not None and i < len(boxes.xyxy):
                bb = boxes.xyxy[i].cpu().numpy().astype(np.float32)
                conf = float(boxes.conf[i].cpu().numpy()) if boxes.conf is not None else 1.0
            else:
                bb = _mask_to_bbox_np(mask_hw)
                conf = 1.0

            bb2d_list.append(bb)
            scores_list.append(conf)

    if len(bb2d_list) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            [],
            np.zeros(0, dtype=np.float32),
        )

    return (
        np.stack(bb2d_list).astype(np.float32),
        masks_list,
        np.array(scores_list, dtype=np.float32),
    )


def _mask_to_bbox_np(mask: np.ndarray) -> np.ndarray:
    """Compute tight AABB from a boolean mask (numpy)."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return np.zeros(4, dtype=np.float32)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def _masks_to_bboxes(masks: np.ndarray) -> np.ndarray:
    """Vectorised mask → bbox for a (M, H, W) bool array."""
    if len(masks) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return np.stack([_mask_to_bbox_np(masks[i]) for i in range(len(masks))])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class SAM3Detector:
    """
    Open-vocabulary text-prompted segmentation using SAM3.

    Attributes
    ----------
    ckpt_dir : str
        Local directory containing SAM3 (and Boxer) weights.
    device : str
        Torch device string ("cuda", "cpu", "mps").

    Notes
    -----
    Weights are auto-downloaded from ``https://huggingface.co/facebook/sam3``
    on first use if the checkpoint directory is empty.
    """

    def __init__(
        self,
        ckpt_dir: Optional[str] = None,
        device: Optional[str] = None,
        auto_download: bool = True,
    ):
        if ckpt_dir is None:
            # Default: ./ckpts relative to project root
            _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ckpt_dir = os.path.join(_root, "ckpts")

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.ckpt_dir = ckpt_dir
        self.device = device

        if auto_download:
            ckpt_dir = _auto_download(ckpt_dir)

        self._backend = _load_sam3_model(ckpt_dir, device)
        print(
            f"[SAM3] Loaded ({self._backend.__class__.__name__}) on {device}"
        )

    # ------------------------------------------------------------------
    # Image-level detection
    # ------------------------------------------------------------------

    def detect(
        self,
        img_np: np.ndarray,
        text_prompts: List[str],
        min_score: float = 0.0,
    ) -> Tuple[torch.Tensor, List[np.ndarray], torch.Tensor]:
        """Run text-prompted segmentation on a single RGB image.

        Args:
            img_np:        (H, W, 3) uint8 RGB numpy array.
            text_prompts:  List of text concepts, e.g. ["coral", "algae"].
            min_score:     Drop detections below this confidence (0 = keep all).

        Returns:
            bb2d:   (M, 4) float32 torch.Tensor  [x0, y0, x1, y1]
            masks:  List of M (H, W) bool numpy arrays
            scores: (M,) float32 torch.Tensor
        """
        bb2d_np, masks, scores_np = self._backend.predict_image(img_np, text_prompts)

        if len(bb2d_np) > 0 and min_score > 0.0:
            keep = scores_np >= min_score
            bb2d_np = bb2d_np[keep]
            masks = [m for m, k in zip(masks, keep) if k]
            scores_np = scores_np[keep]

        bb2d = torch.from_numpy(bb2d_np).float()
        scores = torch.from_numpy(scores_np).float()
        return bb2d, masks, scores

    # ------------------------------------------------------------------
    # Video-level detection (temporally consistent IDs)
    # ------------------------------------------------------------------

    def detect_video(
        self,
        frames: List[np.ndarray],
        text_prompts: List[str],
        min_score: float = 0.0,
    ) -> List[Tuple[torch.Tensor, List[np.ndarray], torch.Tensor]]:
        """Run text-prompted segmentation + tracking across a list of frames.

        Args:
            frames:        List of (H, W, 3) uint8 RGB numpy arrays.
            text_prompts:  List of text concepts.
            min_score:     Drop detections below this confidence.

        Returns:
            List of (bb2d, masks, scores) tuples, one per input frame.
            bb2d:   (M, 4) float32 torch.Tensor  [x0, y0, x1, y1]
            masks:  List of M (H, W) bool numpy arrays
            scores: (M,) float32 torch.Tensor
        """
        raw_results = self._backend.predict_video(frames, text_prompts)
        out = []
        for bb2d_np, masks, scores_np in raw_results:
            if len(bb2d_np) > 0 and min_score > 0.0:
                keep = scores_np >= min_score
                bb2d_np = bb2d_np[keep]
                masks = [m for m, k in zip(masks, keep) if k]
                scores_np = scores_np[keep]
            bb2d = torch.from_numpy(bb2d_np).float()
            scores = torch.from_numpy(scores_np).float()
            out.append((bb2d, masks, scores))
        return out
