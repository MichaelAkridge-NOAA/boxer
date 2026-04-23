#! /usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
run_boxer_sfm.py – Boxer 3-D detection entry point for Metashape SfM data.

Pipeline
--------
  MetashapeLoader → SAM3Detector → (MaskPointFilter) → BoxerNet → fuse_3d_boxes

Quick start
-----------
  # With GPU:
  python run_boxer_sfm.py \\
      --metashape_dir /data/metashape_export \\
      --prompts "coral" \\
      --fuse

  # Specify multiple benthic classes:
  python run_boxer_sfm.py \\
      --metashape_dir /data/export \\
      --prompts "coral,algae,rubble,sand" \\
      --sam3_mode video \\
      --fuse

  # CPU-only (slow but functional):
  python run_boxer_sfm.py --metashape_dir /data/export --force_cpu
"""

import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from boxernet.boxernet import BoxerNet
from loaders.metashape_loader import MetashapeLoader
from sam3.sam3_detector import SAM3Detector
from utils.demo_utils import CKPT_PATH, EVAL_PATH, CudaTimer
from utils.file_io import ObbCsvWriter2
from utils.mask_utils import (
    export_instance_ply,
    filter_sdp_by_mask,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Boxer 3-D detection on Agisoft Metashape SfM exports via SAM3."
    )

    # --- Input / data ---
    parser.add_argument(
        "--metashape_dir",
        type=str,
        required=True,
        help="Root folder of the Metashape export (must contain cameras.xml).",
    )
    parser.add_argument(
        "--xml_filename",
        type=str,
        default="cameras.xml",
        help="Metashape cameras XML filename (default: cameras.xml).",
    )
    parser.add_argument(
        "--image_subdir",
        type=str,
        default=None,
        help="Sub-folder inside --metashape_dir containing images (auto-detected if omitted).",
    )
    parser.add_argument(
        "--ply_filename",
        type=str,
        default="dense.ply",
        help="Dense point-cloud PLY filename (default: dense.ply).",
    )
    parser.add_argument(
        "--sdp_radius",
        type=float,
        default=5.0,
        help="KD-tree radius (metres) for per-frame PLY point sampling (default: 5.0).",
    )
    parser.add_argument(
        "--gravity_vec",
        type=float,
        nargs=3,
        default=None,
        metavar=("GX", "GY", "GZ"),
        help="Gravity direction in world space (default: 0 0 -1 for Z-up).",
    )
    parser.add_argument("--skip_n", type=int, default=1, help="Skip every N frames.")
    parser.add_argument("--start_n", type=int, default=0, help="Start at frame index N.")
    parser.add_argument("--max_n", type=int, default=99999, help="Max frames to process.")

    # --- SAM3 ---
    parser.add_argument(
        "--prompts",
        type=str,
        default="coral",
        help='Comma-separated SAM3 text prompts (default: "coral").',
    )
    parser.add_argument(
        "--sam3_ckpt",
        type=str,
        default=os.path.join(CKPT_PATH),
        help=f"Directory containing SAM3 weights (default: {CKPT_PATH}). "
             "Auto-downloaded from HuggingFace if weights are absent.",
    )
    parser.add_argument(
        "--sam3_mode",
        type=str,
        default="image",
        choices=["image", "video"],
        help='SAM3 inference mode.  "video" uses temporal tracking for consistent '
             "instance IDs across frames (default: image).",
    )
    parser.add_argument(
        "--thresh2d",
        type=float,
        default=0.1,
        help="Minimum SAM3 segmentation confidence to keep a detection (default: 0.1).",
    )

    # --- Boxer ---
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to BoxerNet checkpoint.  Auto-resolved from --sam3_ckpt if omitted.",
    )
    parser.add_argument(
        "--thresh3d",
        type=float,
        default=0.1,
        help="Minimum BoxerNet 3-D confidence to keep a detection (default: 0.1). "
             "Lower values recommended for coral (model trained on indoor objects).",
    )
    parser.add_argument(
        "--no_mask_filter",
        action="store_true",
        help="Disable SAM3-mask-based SDP filtering (use full point cloud per detection).",
    )

    # --- Post-processing ---
    parser.add_argument("--fuse", action="store_true", help="Run offline 3-D box fusion.")
    parser.add_argument(
        "--export_ply",
        action="store_true",
        help="Export coloured per-instance point cloud to coral_instances.ply.",
    )

    # --- Output ---
    parser.add_argument(
        "--output_dir",
        type=str,
        default=EVAL_PATH,
        help=f"Output directory (default: {EVAL_PATH}).",
    )
    parser.add_argument(
        "--write_name",
        type=str,
        default="boxer_sfm",
        help="Prefix for output files (default: boxer_sfm).",
    )
    parser.add_argument(
        "--save_masks",
        action="store_true",
        help="Save per-frame SAM3 mask overlays as PNG images.",
    )
    parser.add_argument("--no_csv", action="store_true", help="Skip CSV writing.")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU inference.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_boxer_ckpt(sam3_ckpt_dir: str, explicit_ckpt: str | None) -> str:
    """Find a BoxerNet checkpoint."""
    if explicit_ckpt and os.path.exists(explicit_ckpt):
        return explicit_ckpt

    # Search sam3_ckpt_dir for any .ckpt file with "boxernet" in the name
    if os.path.isdir(sam3_ckpt_dir):
        for fname in os.listdir(sam3_ckpt_dir):
            if fname.endswith(".ckpt") and "boxernet" in fname.lower():
                return os.path.join(sam3_ckpt_dir, fname)
        # Fallback: any .ckpt file
        for fname in os.listdir(sam3_ckpt_dir):
            if fname.endswith(".ckpt"):
                return os.path.join(sam3_ckpt_dir, fname)

    # Try the default ckpts folder
    default_dir = CKPT_PATH
    if os.path.isdir(default_dir):
        for fname in os.listdir(default_dir):
            if fname.endswith(".ckpt"):
                return os.path.join(default_dir, fname)

    raise FileNotFoundError(
        "Could not find a BoxerNet checkpoint.  "
        "Pass --ckpt /path/to/boxernet.ckpt or run scripts/download_weights.sh."
    )


def _save_mask_overlay(
    img_np: np.ndarray,
    masks: list,
    output_path: str,
    alpha: float = 0.45,
) -> None:
    """Save a debug PNG showing all SAM3 masks overlaid on the image."""
    overlay = img_np.copy()
    colours = [
        (255, 80, 80),
        (80, 255, 80),
        (80, 80, 255),
        (255, 255, 80),
        (255, 80, 255),
        (80, 255, 255),
    ]
    for i, mask in enumerate(masks):
        colour = colours[i % len(colours)]
        mask_u8 = mask.astype(np.uint8)
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_u8 > 0,
                np.clip(overlay[:, :, c] * (1 - alpha) + colour[c] * alpha, 0, 255).astype(np.uint8),
                overlay[:, :, c],
            )
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = _parse_args()

    # ------------------------------------------------------------------ device
    if args.force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"==> Device: {device}")

    # ------------------------------------------------------------------ paths
    seq_name = os.path.basename(args.metashape_dir.rstrip("/"))
    output_dir = os.path.expanduser(args.output_dir)
    log_dir = os.path.join(output_dir, seq_name)
    os.makedirs(log_dir, exist_ok=True)

    csv_path = os.path.join(log_dir, f"{args.write_name}_3dbbs.csv")
    masks_dir = os.path.join(log_dir, "sam3_masks") if args.save_masks else None
    if masks_dir:
        os.makedirs(masks_dir, exist_ok=True)

    print(f"==> Output: {log_dir}")

    # ----------------------------------------------------------- parse prompts
    text_prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    print(f"==> SAM3 prompts: {text_prompts}")

    # ------------------------------------------------------------------ loader
    loader = MetashapeLoader(
        metashape_dir=args.metashape_dir,
        xml_filename=args.xml_filename,
        image_subdir=args.image_subdir,
        ply_filename=args.ply_filename,
        sdp_radius=args.sdp_radius,
        gravity_vec=args.gravity_vec,
        skip_frames=args.skip_n,
        max_frames=args.max_n,
        start_frame=args.start_n,
    )

    # ------------------------------------------------------------------ SAM3
    sam3 = SAM3Detector(
        ckpt_dir=args.sam3_ckpt,
        device=device,
    )

    # --------------------------------------------------------------- BoxerNet
    boxer_ckpt = _resolve_boxer_ckpt(args.sam3_ckpt, args.ckpt)
    print(f"==> BoxerNet checkpoint: {boxer_ckpt}")
    boxernet = BoxerNet.load_from_checkpoint(boxer_ckpt, device=device)

    # Set loader resize to match BoxerNet expected input size
    loader.resize = boxernet.hw
    loader.index = 0
    loader._init_prefetch()
    print(f"==> Resizing images to {loader.resize}×{loader.resize} for BoxerNet")

    # ------------------------------------------------------------------ CSV writer
    writer = None if args.no_csv else ObbCsvWriter2(csv_path)

    # ------------------------------------------------------------------ video mode pre-fetch
    if args.sam3_mode == "video":
        print("==> Pre-loading frames for SAM3 video mode …")
        all_frames = []
        all_datums = []
        for datum in loader:
            img_t = datum["img0"]  # (1, 3, H, W) float [0,1]
            img_np = (img_t[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            all_frames.append(img_np)
            all_datums.append(datum)
        loader.index = 0  # reset (prefetch already done above)

        print(f"==> Running SAM3 video mode on {len(all_frames)} frames …")
        video_results = sam3.detect_video(
            all_frames, text_prompts, min_score=args.thresh2d
        )
    else:
        video_results = None

    # ------------------------------------------------------------------ per-frame instance store (for PLY export)
    instance_sdp: dict = {}  # label → list of (N,3) point tensors

    # ------------------------------------------------------------------ main loop
    timer = CudaTimer(device)
    pbar = tqdm(
        range(len(loader) if video_results is None else len(all_datums)),
        desc="run_boxer_sfm",
    )

    for ii in pbar:
        timer.start("frame")

        # ----- Get datum -----
        if video_results is not None:
            datum = all_datums[ii]
            img_t = datum["img0"]
            img_np = (img_t[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            bb2d_t, masks, scores = video_results[ii]
        else:
            try:
                datum = next(loader)
            except StopIteration:
                break
            img_t = datum["img0"]
            img_np = (img_t[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            bb2d_t, masks, scores = sam3.detect(
                img_np, text_prompts, min_score=args.thresh2d
            )

        cam = datum["cam0"]
        T_world_rig = datum["T_world_rig0"]
        sdp_w = datum["sdp_w"]
        time_ns = datum.get("time_ns0", ii)

        # ----- Save mask overlays -----
        if masks_dir and len(masks) > 0:
            overlay_path = os.path.join(masks_dir, f"frame_{ii:05d}.png")
            _save_mask_overlay(img_np, masks, overlay_path)

        if bb2d_t.shape[0] == 0:
            pbar.set_postfix_str("no detections")
            continue

        # ----- Optionally filter SDP by mask -----
        if not args.no_mask_filter and len(masks) == bb2d_t.shape[0]:
            per_det_sdp = []
            for det_idx, mask in enumerate(masks):
                filtered = filter_sdp_by_mask(
                    sdp_w, mask, cam, T_world_rig, min_points=10
                )
                per_det_sdp.append(filtered)
        else:
            per_det_sdp = [sdp_w] * bb2d_t.shape[0]

        # ----- Run BoxerNet per detection -----
        timer.start("boxernet")
        obb_list = []
        for det_idx in range(bb2d_t.shape[0]):
            bb2d_single = bb2d_t[det_idx : det_idx + 1]  # (1, 4)
            sdp_single = per_det_sdp[det_idx]

            # Assign the SAM3 prompt label to the detection
            # (BoxerNet uses sem_id internally; we pass it through text)
            label = text_prompts[det_idx % len(text_prompts)]

            try:
                with torch.no_grad():
                    obbs = boxernet.forward(
                        img=img_t.to(device),
                        cam=cam,
                        T_world_rig=T_world_rig,
                        sdp_w=sdp_single,
                        bb2d=bb2d_single.to(device),
                    )
            except Exception as exc:
                # Individual detection failure should not abort the whole frame
                print(f"  [warn] BoxerNet failed for frame {ii}, det {det_idx}: {exc}")
                continue

            # Filter by 3D confidence
            if len(obbs) > 0:
                keep = obbs.prob.squeeze(-1) >= args.thresh3d
                obbs = obbs[keep]

            if len(obbs) > 0:
                obb_list.append(obbs)

                # Accumulate points for PLY export
                if args.export_ply:
                    pts = per_det_sdp[det_idx]
                    if pts is not None and len(pts) > 0:
                        instance_sdp.setdefault(label, []).append(pts)

        timer.stop("boxernet")

        # ----- Merge & write -----
        if len(obb_list) > 0:
            from utils.tw.obb import ObbTW

            all_obbs = ObbTW(
                torch.cat([o._data for o in obb_list], dim=0)
            )
            if writer is not None:
                writer.write(time_ns=time_ns, obbs=all_obbs)

        timer.stop("frame")
        pbar.set_postfix_str(
            f"dets={bb2d_t.shape[0]} "
            f"sam3={timer.get_ms('frame'):.0f}ms"
        )

    # ------------------------------------------------------------------ finalise
    if writer is not None:
        writer.close()
        print(f"==> 3D detections written to {csv_path}")

    # ------------------------------------------------------------------ fusion
    if args.fuse:
        from utils.fuse_3d_boxes import fuse_obbs_from_csv

        print(f"\n==> Running 3-D box fusion on {csv_path}")
        fuse_obbs_from_csv(csv_path)

    # ------------------------------------------------------------------ PLY export
    if args.export_ply and instance_sdp:
        ply_path = os.path.join(log_dir, "coral_instances.ply")
        instances = [
            (torch.cat(pts_list, dim=0), label)
            for label, pts_list in instance_sdp.items()
        ]
        export_instance_ply(instances, ply_path)

    print("==> Done.")


if __name__ == "__main__":
    main()
