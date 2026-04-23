# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Data loader for Agisoft Metashape SfM exports.

Expects an export folder with:
  cameras.xml   – Agisoft XML with per-camera intrinsics + extrinsics
  images/       – per-frame RGB images (jpg/png; subfolder name is configurable)
  dense.ply     – optional world-frame dense point cloud
  depth_maps/   – optional per-image depth maps (EXR or TIFF, float metres)

The loader satisfies the BaseLoader contract so it plugs directly into the
existing run_boxer.py / run_boxer_sfm.py inference pipelines.

Coordinate conventions
----------------------
Metashape stores camera *transform* matrices as T_world_cam (4×4, world-from-
camera), which matches Boxer's convention.  We parse them directly and build
PoseTW from the rotation and translation blocks.

Gravity
-------
Defaults to Z-up (gravity direction [0, 0, -1] in world space).  Pass an
explicit ``gravity_vec`` tuple/list to override (e.g. [0, -1, 0] for Y-up).
"""

import os
from typing import List, Optional, Tuple
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import torch

from loaders.base_loader import BaseLoader
from utils.tw.obb import ObbTW
from utils.tw.pose import PoseTW


def _parse_matrix(text: str) -> np.ndarray:
    """Parse a Metashape space/newline-separated 4×4 matrix string."""
    vals = [float(v) for v in text.split()]
    assert len(vals) == 16, f"Expected 16 values for 4x4 matrix, got {len(vals)}"
    return np.array(vals, dtype=np.float64).reshape(4, 4)


def _parse_vec(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.split()], dtype=np.float64)


class MetashapeCamera:
    """Per-camera metadata parsed from cameras.xml."""

    __slots__ = ("image_path", "w", "h", "fx", "fy", "cx", "cy", "T_world_cam")

    def __init__(self, image_path, w, h, fx, fy, cx, cy, T_world_cam):
        self.image_path = image_path  # absolute path to RGB image
        self.w = w
        self.h = h
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.T_world_cam: np.ndarray = T_world_cam  # (4, 4) float64 world-from-cam


def parse_cameras_xml(
    xml_path: str,
    image_root: Optional[str] = None,
) -> List[MetashapeCamera]:
    """Parse Agisoft cameras.xml and return a list of MetashapeCamera objects.

    Args:
        xml_path: Path to cameras.xml exported by Agisoft Metashape.
        image_root: Directory that contains the raw images.  If None, uses the
            directory containing cameras.xml as a fallback search root.

    Returns:
        List of MetashapeCamera (one per active camera with a valid transform).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Resolve image root
    xml_dir = os.path.dirname(os.path.abspath(xml_path))
    if image_root is None:
        # Try common sub-folder names alongside cameras.xml
        for candidate in ("images", "raw_images", "frames", "photos", "."):
            candidate_path = os.path.join(xml_dir, candidate)
            if os.path.isdir(candidate_path):
                image_root = candidate_path
                break
        if image_root is None:
            image_root = xml_dir

    cameras: List[MetashapeCamera] = []

    # Agisoft XML structure:
    # <document> <chunk> <sensors> <sensor id="0"> <calibration> ...
    #            <cameras> <camera id="0" sensor_id="0" label="img.jpg">
    #                        <transform>...</transform>
    # We need to link sensors → calibrations and cameras → transforms.

    chunk = root.find("chunk")
    if chunk is None:
        chunk = root  # some exports omit the chunk wrapper

    # --- Build sensor id → calibration ---
    sensor_cals: dict = {}  # sensor_id -> (w, h, fx, fy, cx, cy)
    sensors_el = chunk.find("sensors")
    if sensors_el is not None:
        for sensor in sensors_el.findall("sensor"):
            sid = sensor.get("id", "0")
            cal = sensor.find("calibration")
            if cal is None:
                continue
            # resolution may be on sensor or calibration
            res_el = sensor.find("resolution") or cal.find("resolution")
            if res_el is None:
                continue
            w = int(res_el.get("width", 0))
            h = int(res_el.get("height", 0))

            # focal length: f (square pixel) or fx/fy
            fx_el = cal.find("fx")
            fy_el = cal.find("fy")
            f_el = cal.find("f")
            if fx_el is not None and fy_el is not None:
                fx = float(fx_el.text)
                fy = float(fy_el.text)
            elif f_el is not None:
                fx = fy = float(f_el.text)
            else:
                # fallback: use image width as proxy
                fx = fy = float(max(w, h))

            cx_el = cal.find("cx")
            cy_el = cal.find("cy")
            cx = float(cx_el.text) if cx_el is not None else 0.0
            cy = float(cy_el.text) if cy_el is not None else 0.0
            # Metashape stores cx/cy as offsets from image centre
            cx = w / 2.0 + cx
            cy = h / 2.0 + cy

            sensor_cals[sid] = (w, h, fx, fy, cx, cy)

    # --- Build cameras ---
    cameras_el = chunk.find("cameras")
    if cameras_el is None:
        raise ValueError(f"No <cameras> element found in {xml_path}")

    for cam_el in cameras_el.findall("camera"):
        # Skip disabled / reference cameras
        if cam_el.get("enabled", "true").lower() == "false":
            continue

        transform_el = cam_el.find("transform")
        if transform_el is None:
            continue  # camera without pose (not yet aligned)

        T_world_cam = _parse_matrix(transform_el.text.strip())

        label = cam_el.get("label", "")
        sensor_id = cam_el.get("sensor_id", "0")

        if sensor_id not in sensor_cals:
            # Sensor not found: skip
            continue

        w, h, fx, fy, cx, cy = sensor_cals[sensor_id]

        # Locate image file
        image_path = _find_image(label, image_root, xml_dir)

        cameras.append(
            MetashapeCamera(
                image_path=image_path,
                w=w,
                h=h,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                T_world_cam=T_world_cam,
            )
        )

    return cameras


def _find_image(label: str, image_root: str, xml_dir: str) -> str:
    """Try several strategies to locate an image file given its Metashape label."""
    # Strategy 1: label is an absolute path that exists
    if os.path.isabs(label) and os.path.exists(label):
        return label

    # Strategy 2: label relative to image_root
    candidate = os.path.join(image_root, label)
    if os.path.exists(candidate):
        return candidate

    # Strategy 3: basename of label in image_root (handles sub-path labels)
    basename = os.path.basename(label)
    candidate = os.path.join(image_root, basename)
    if os.path.exists(candidate):
        return candidate

    # Strategy 4: search xml_dir recursively for the basename
    for dirpath, _, filenames in os.walk(xml_dir):
        if basename in filenames:
            return os.path.join(dirpath, basename)

    # Not found: return the best-effort path (caller should handle missing files)
    return candidate


# ---------------------------------------------------------------------------
# MetashapeLoader
# ---------------------------------------------------------------------------


class MetashapeLoader(BaseLoader):
    """
    Data loader for Agisoft Metashape SfM exports.

    Yields datums conforming to BaseLoader contract, ready for BoxerNet.
    """

    camera: str = "metashape"
    device_name: str = "Metashape"

    def __init__(
        self,
        metashape_dir: str,
        xml_filename: str = "cameras.xml",
        image_subdir: Optional[str] = None,
        ply_filename: str = "dense.ply",
        depth_subdir: str = "depth_maps",
        skip_frames: int = 1,
        max_frames: Optional[int] = None,
        start_frame: int = 0,
        sdp_radius: float = 5.0,
        gravity_vec: Optional[Tuple[float, float, float]] = None,
    ):
        """
        Args:
            metashape_dir: Root folder of the Metashape export.
            xml_filename: Name of the cameras XML file (default: cameras.xml).
            image_subdir: Sub-folder containing images.  Auto-detected if None.
            ply_filename: Name of the dense point-cloud PLY (default: dense.ply).
            depth_subdir: Sub-folder containing per-image depth maps.
            skip_frames: Process every N-th frame.
            max_frames: Hard cap on number of frames (None = no limit).
            start_frame: 0-based index of first frame to use.
            sdp_radius: Radius (metres) for KD-tree per-frame point query.
            gravity_vec: Gravity direction in world space (default: Z-up [0,0,-1]).
        """
        metashape_dir = os.path.expanduser(metashape_dir)
        self.metashape_dir = metashape_dir
        self.sdp_radius = sdp_radius
        self.gravity_vec = np.array(
            gravity_vec if gravity_vec is not None else [0.0, 0.0, -1.0],
            dtype=np.float32,
        )

        # --- Parse cameras.xml ---
        xml_path = os.path.join(metashape_dir, xml_filename)
        image_root = (
            os.path.join(metashape_dir, image_subdir)
            if image_subdir is not None
            else None
        )
        all_cams = parse_cameras_xml(xml_path, image_root=image_root)

        # Apply start/skip/max
        all_cams = all_cams[start_frame::skip_frames]
        if max_frames is not None:
            all_cams = all_cams[:max_frames]

        # Filter cameras whose image file does not exist
        valid_cams = []
        for cam in all_cams:
            if cam.image_path and os.path.exists(cam.image_path):
                valid_cams.append(cam)
            else:
                print(
                    f"MetashapeLoader: skipping camera '{os.path.basename(cam.image_path or '')}' "
                    f"– image not found at {cam.image_path}"
                )

        if len(valid_cams) == 0:
            raise ValueError(
                f"No valid cameras found in {xml_path}. "
                "Check that image files exist and paths in cameras.xml are correct."
            )

        self.cams: List[MetashapeCamera] = valid_cams
        self.length = len(self.cams)
        self.index = 0
        self.resize: Optional[int] = None  # set externally by BoxerNetWrapper

        # --- Recenter around first camera ---
        self.world_offset = self.cams[0].T_world_cam[:3, 3].copy().astype(np.float32)

        # --- Load dense PLY (optional) ---
        self._kdtree = None
        self._world_pts = None
        ply_path = os.path.join(metashape_dir, ply_filename)
        if os.path.exists(ply_path):
            self._load_ply(ply_path)
        else:
            print(
                f"MetashapeLoader: dense PLY not found at {ply_path}. "
                "SDP will fall back to depth maps (if available)."
            )

        # --- Depth map directory (optional) ---
        self.depth_dir = os.path.join(metashape_dir, depth_subdir)
        if not os.path.isdir(self.depth_dir):
            self.depth_dir = None

        print(
            f"MetashapeLoader: {self.length} cameras, "
            f"PLY={'yes' if self._kdtree is not None else 'no'}, "
            f"depth={'yes' if self.depth_dir else 'no'}"
        )

        self._init_prefetch()

    # ------------------------------------------------------------------
    # PLY helpers
    # ------------------------------------------------------------------

    def _load_ply(self, ply_path: str):
        """Load a dense PLY and build a KD-tree for per-frame radius queries."""
        try:
            pts = _read_ply_xyz(ply_path)
        except Exception as e:
            print(f"MetashapeLoader: failed to load PLY ({e}); SDP disabled.")
            return

        # Recenter
        pts -= self.world_offset[None, :]

        self._world_pts = pts.astype(np.float32)
        try:
            from scipy.spatial import KDTree

            self._kdtree = KDTree(self._world_pts)
            print(
                f"MetashapeLoader: KD-tree built from {len(self._world_pts):,} PLY points"
            )
        except ImportError:
            print(
                "MetashapeLoader: scipy not available – SDP will sample uniformly from PLY."
            )

    def _sdp_from_ply(
        self,
        cam_pos_w: np.ndarray,
        num_samples: int = 10_000,
    ) -> torch.Tensor:
        """Return up to num_samples world-frame points near cam_pos_w."""
        if self._world_pts is None:
            return torch.zeros(0, 3, dtype=torch.float32)

        if self._kdtree is not None:
            indices = self._kdtree.query_ball_point(
                cam_pos_w, r=self.sdp_radius, return_sorted=False
            )
            if len(indices) == 0:
                return torch.zeros(0, 3, dtype=torch.float32)
            pts = self._world_pts[indices]
        else:
            pts = self._world_pts

        if len(pts) > num_samples:
            idx = np.random.choice(len(pts), size=num_samples, replace=False)
            pts = pts[idx]

        return torch.from_numpy(pts)

    # ------------------------------------------------------------------
    # BaseLoader protocol
    # ------------------------------------------------------------------

    def load(self, idx: int) -> dict:
        mc = self.cams[idx]

        # --- Load image ---
        img_bgr = cv2.imread(mc.image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {mc.image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        HH, WW = img_rgb.shape[:2]

        # --- Resize ---
        if self.resize is not None:
            resizeH = resizeW = self.resize
            scale_x = resizeW / WW
            scale_y = resizeH / HH
            img_rgb = cv2.resize(
                img_rgb, (resizeW, resizeH), interpolation=cv2.INTER_LINEAR
            )
        else:
            resizeH, resizeW = HH, WW
            scale_x = scale_y = 1.0

        fx = mc.fx * scale_x
        fy = mc.fy * scale_y
        cx = mc.cx * scale_x
        cy = mc.cy * scale_y

        datum: dict = {}
        datum["img0"] = self.img_to_tensor(img_rgb)

        # --- Camera ---
        cam = self.pinhole_from_K(
            resizeW, resizeH, fx, fy, cx, cy,
            valid_radius=(resizeW, resizeH),
        )
        datum["cam0"] = cam.float()

        # --- Pose (recenter) ---
        T_wc = mc.T_world_cam.copy().astype(np.float32)
        T_wc[:3, 3] -= self.world_offset
        R_flat = T_wc[:3, :3].flatten()
        t_vec = T_wc[:3, 3]
        datum["T_world_rig0"] = PoseTW(
            torch.tensor([*R_flat, *t_vec], dtype=torch.float32)
        )

        # --- SDP ---
        cam_pos_w = T_wc[:3, 3]
        if self._kdtree is not None or self._world_pts is not None:
            datum["sdp_w"] = self._sdp_from_ply(cam_pos_w)
        else:
            # Try depth map fallback
            depth_np = self._load_depth(mc, resizeH, resizeW)
            T_wc_full = mc.T_world_cam.copy().astype(np.float32)
            T_wc_full[:3, 3] -= self.world_offset
            datum["sdp_w"] = self.sdp_from_depth(
                depth_np, fx, fy, cx, cy,
                T_wc_full[:3, :3], T_wc_full[:3, 3],
            )

        # --- Metadata ---
        datum["time_ns0"] = idx
        datum["rotated0"] = torch.tensor(False).reshape(1)
        datum["bb2d0"] = torch.zeros(0, 4, dtype=torch.float32)
        datum["obbs"] = ObbTW(torch.zeros(0, 165))
        datum["gt_labels"] = []

        return datum

    def _load_depth(
        self,
        mc: MetashapeCamera,
        resizeH: int,
        resizeW: int,
    ) -> Optional[np.ndarray]:
        """Attempt to load a depth map matching mc.image_path from depth_dir."""
        if self.depth_dir is None:
            return None

        stem = os.path.splitext(os.path.basename(mc.image_path))[0]
        for ext in (".exr", ".tiff", ".tif", ".png"):
            depth_path = os.path.join(self.depth_dir, stem + ext)
            if os.path.exists(depth_path):
                if ext == ".exr":
                    depth_np = cv2.imread(
                        depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
                    )
                elif ext in (".tiff", ".tif"):
                    depth_np = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                else:  # png (uint16 mm)
                    raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                    if raw is not None:
                        depth_np = raw.astype(np.float32) / 1000.0
                    else:
                        depth_np = None

                if depth_np is not None and depth_np.ndim == 2:
                    depth_np = depth_np.astype(np.float32)
                    if depth_np.shape[:2] != (resizeH, resizeW):
                        depth_np = cv2.resize(
                            depth_np,
                            (resizeW, resizeH),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    return depth_np

        return None


# ---------------------------------------------------------------------------
# Minimal PLY reader (avoids open3d dependency for basic use)
# ---------------------------------------------------------------------------


def _read_ply_xyz(ply_path: str) -> np.ndarray:
    """Read XYZ coordinates from a PLY file.

    Supports ASCII and binary_little_endian formats.  Only reads x/y/z vertex
    properties; ignores colour, normals, etc.

    Falls back to open3d if available (handles compressed / large files better).
    """
    try:
        import open3d as o3d  # type: ignore

        pcd = o3d.io.read_point_cloud(ply_path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        if len(pts) == 0:
            raise ValueError("open3d returned 0 points")
        return pts
    except ImportError:
        pass

    # --- Pure-Python fallback ---
    with open(ply_path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        # Parse header
        num_vertices = 0
        is_binary_le = False
        prop_names: List[str] = []
        prop_types: List[str] = []
        for line in header_lines:
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            elif line == "format binary_little_endian 1.0":
                is_binary_le = True
            elif line.startswith("property"):
                parts = line.split()
                prop_types.append(parts[1])
                prop_names.append(parts[2])

        _type_map = {
            "float": ("f", 4),
            "float32": ("f", 4),
            "double": ("d", 8),
            "float64": ("d", 8),
            "uchar": ("B", 1),
            "uint8": ("B", 1),
            "int": ("i", 4),
            "int32": ("i", 4),
        }
        row_fmt = ""
        row_size = 0
        for pt in prop_types:
            fmt_char, sz = _type_map.get(pt, ("B", 1))
            row_fmt += fmt_char
            row_size += sz

        x_idx = prop_names.index("x") if "x" in prop_names else None
        y_idx = prop_names.index("y") if "y" in prop_names else None
        z_idx = prop_names.index("z") if "z" in prop_names else None

        if x_idx is None or y_idx is None or z_idx is None:
            raise ValueError("PLY file does not have x/y/z vertex properties")

        if is_binary_le:
            import struct

            raw = f.read(row_size * num_vertices)
            pts = np.zeros((num_vertices, 3), dtype=np.float32)
            for i in range(num_vertices):
                vals = struct.unpack_from(
                    "<" + row_fmt, raw, offset=i * row_size
                )
                pts[i, 0] = vals[x_idx]
                pts[i, 1] = vals[y_idx]
                pts[i, 2] = vals[z_idx]
        else:
            # ASCII
            pts = np.zeros((num_vertices, 3), dtype=np.float32)
            for i in range(num_vertices):
                parts = f.readline().split()
                pts[i, 0] = float(parts[x_idx])
                pts[i, 1] = float(parts[y_idx])
                pts[i, 2] = float(parts[z_idx])

    return pts
