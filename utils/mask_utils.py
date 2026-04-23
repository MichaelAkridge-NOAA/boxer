# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Mask utility functions for the SAM3 + Boxer SfM pipeline.

Provides:
  mask_to_bbox        – convert a boolean pixel mask to a tight AABB
  filter_sdp_by_mask  – keep only world-frame SDP points inside a pixel mask
  export_instance_ply – write a coloured per-instance PLY point cloud
"""

import os
import struct
from typing import List, Optional, Tuple

import numpy as np
import torch

from utils.tw.camera import CameraTW

# ---------------------------------------------------------------------------
# Mask → bounding box
# ---------------------------------------------------------------------------


def mask_to_bbox(mask_hw: np.ndarray) -> Tuple[int, int, int, int]:
    """Return the tight axis-aligned bounding box of a boolean mask.

    Args:
        mask_hw: (H, W) boolean numpy array.

    Returns:
        (x0, y0, x1, y1) integer pixel coordinates (inclusive).
        Returns (0, 0, 0, 0) if the mask is empty.
    """
    rows = np.any(mask_hw, axis=1)
    cols = np.any(mask_hw, axis=0)
    if not rows.any():
        return (0, 0, 0, 0)
    y0, y1 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    x0, x1 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    return (x0, y0, x1, y1)


def mask_to_bbox_tensor(mask_hw: np.ndarray) -> torch.Tensor:
    """Return the tight AABB as a (4,) float32 tensor [x0, y0, x1, y1]."""
    x0, y0, x1, y1 = mask_to_bbox(mask_hw)
    return torch.tensor([x0, y0, x1, y1], dtype=torch.float32)


# ---------------------------------------------------------------------------
# SDP point-cloud mask filtering
# ---------------------------------------------------------------------------


def filter_sdp_by_mask(
    sdp_w: torch.Tensor,
    mask_hw: np.ndarray,
    cam: "CameraTW",
    T_world_cam: torch.Tensor,
    min_points: int = 10,
) -> torch.Tensor:
    """Return the subset of world-frame SDP points that project inside *mask_hw*.

    Projects each 3-D world point through the camera model and retains only
    those whose 2-D projection falls within the segmentation mask.

    Args:
        sdp_w:       (N, 3) float32 world-frame semi-dense points.
        mask_hw:     (H, W) boolean numpy mask (SAM3 output).
        cam:         CameraTW camera model for the current frame.
        T_world_cam: (12,) or (4, 4) tensor giving the world-from-camera pose
                     used to invert back to camera frame.
        min_points:  If fewer than this many points survive, return the full
                     sdp_w unfiltered (avoids degenerate empty OBBs).

    Returns:
        Filtered (K, 3) float32 tensor.  May be the original sdp_w if too few
        points are inside the mask.
    """
    if sdp_w is None or sdp_w.shape[0] == 0:
        return sdp_w

    mask_t = torch.from_numpy(mask_hw.astype(np.uint8))  # (H, W) uint8
    H, W = mask_t.shape

    # Build camera-from-world transform
    T_wc = _pose_to_4x4(T_world_cam)
    T_cw = torch.linalg.inv(T_wc.double()).float()  # (4, 4)

    # Transform world points → camera frame
    pts_w_h = torch.cat(
        [sdp_w, torch.ones(sdp_w.shape[0], 1, dtype=torch.float32)], dim=1
    )  # (N, 4)
    pts_c = (T_cw @ pts_w_h.T).T[:, :3]  # (N, 3)

    # Project to pixel coordinates using a simple pinhole model extracted from cam
    # CameraTW stores: [w, h, fx, fy, cx, cy, model_id, near, vr_w, vr_h, R_flat(9), t(3)]
    cam_data = cam._data
    fx = cam_data[2].item()
    fy = cam_data[3].item()
    cx = cam_data[4].item()
    cy = cam_data[5].item()

    # Only project points in front of camera (z > 0)
    z = pts_c[:, 2]
    valid_z = z > 1e-3
    px = torch.full((pts_c.shape[0],), -1.0)
    py = torch.full((pts_c.shape[0],), -1.0)
    px[valid_z] = fx * pts_c[valid_z, 0] / pts_c[valid_z, 2] + cx
    py[valid_z] = fy * pts_c[valid_z, 1] / pts_c[valid_z, 2] + cy

    # Check which projected pixels fall inside the mask
    pxi = px.long()
    pyi = py.long()
    in_bounds = valid_z & (pxi >= 0) & (pxi < W) & (pyi >= 0) & (pyi < H)

    # Vectorised mask lookup
    inside = torch.zeros(sdp_w.shape[0], dtype=torch.bool)
    ib_idx = torch.where(in_bounds)[0]
    if ib_idx.numel() > 0:
        inside[ib_idx] = mask_t[pyi[ib_idx], pxi[ib_idx]].bool()

    n_inside = inside.sum().item()
    if n_inside < min_points:
        # Too few points in mask – return full cloud to avoid degenerate OBB
        return sdp_w

    return sdp_w[inside]


def _pose_to_4x4(pose) -> torch.Tensor:
    """Convert a PoseTW / (12,) tensor / (4,4) tensor to a (4,4) float32 matrix."""
    if isinstance(pose, torch.Tensor):
        if pose.shape == torch.Size([4, 4]):
            return pose.float()
        if pose.numel() == 12:
            # [R_flat(9), t(3)] convention used by PoseTW
            data = pose.float().flatten()
            R = data[:9].reshape(3, 3)
            t = data[9:12]
            M = torch.eye(4, dtype=torch.float32)
            M[:3, :3] = R
            M[:3, 3] = t
            return M
    # PoseTW
    data = pose._data.float().flatten()
    R = data[:9].reshape(3, 3)
    t = data[9:12]
    M = torch.eye(4, dtype=torch.float32)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


# ---------------------------------------------------------------------------
# Coloured per-instance PLY export
# ---------------------------------------------------------------------------

# Palette: TAB20-inspired RGB colours (uint8)
_INSTANCE_PALETTE = np.array(
    [
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229],
    ],
    dtype=np.uint8,
)


def export_instance_ply(
    instances: List[Tuple[torch.Tensor, str]],
    output_path: str,
    palette: Optional[np.ndarray] = None,
) -> None:
    """Write a coloured point cloud PLY with one colour per instance.

    Args:
        instances: List of (points_Nx3, label_string) tuples.
        output_path: Destination `.ply` file path.
        palette: (K, 3) uint8 RGB palette.  Cycles through default palette if None.
    """
    if palette is None:
        palette = _INSTANCE_PALETTE

    all_pts: List[np.ndarray] = []
    all_rgb: List[np.ndarray] = []

    for inst_idx, (pts_tensor, _label) in enumerate(instances):
        pts = pts_tensor.cpu().numpy().astype(np.float32)
        # Filter NaN / Inf
        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        if len(pts) == 0:
            continue
        colour = palette[inst_idx % len(palette)]
        rgb = np.tile(colour[None, :], (len(pts), 1))
        all_pts.append(pts)
        all_rgb.append(rgb)

    if len(all_pts) == 0:
        # Write empty PLY
        _write_ply(output_path, np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8))
        return

    pts_all = np.concatenate(all_pts, axis=0)
    rgb_all = np.concatenate(all_rgb, axis=0)
    _write_ply(output_path, pts_all, rgb_all)
    print(f"[mask_utils] Wrote {len(pts_all):,} points to {output_path}")


def _write_ply(path: str, pts: np.ndarray, rgb: np.ndarray) -> None:
    """Write a binary-little-endian PLY with x/y/z/red/green/blue."""
    n = len(pts)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(struct.pack("<fffBBB", pts[i, 0], pts[i, 1], pts[i, 2],
                                int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2])))
