# Boxer SfM Deployment Guide

Step-by-step instructions for running the **Boxer + SAM3 coral detection pipeline** on Agisoft Metashape SfM exports — from a fresh machine to labelled 3D bounding boxes and coloured point clouds.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Repository Setup](#2-repository-setup)
3. [Metashape Export Format](#3-metashape-export-format)
4. [Option A — Docker Compose (recommended)](#4-option-a--docker-compose-recommended)
   - 4.1 [GPU deployment](#41-gpu-deployment)
   - 4.2 [CPU-only deployment](#42-cpu-only-deployment)
5. [Option B — Direct Python (no Docker)](#5-option-b--direct-python-no-docker)
6. [CLI Reference](#6-cli-reference)
7. [Understanding the Outputs](#7-understanding-the-outputs)
8. [SAM3 Prompt Tuning](#8-sam3-prompt-tuning)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prerequisites

### Hardware

| Scenario | Minimum | Recommended |
|---|---|---|
| Docker GPU | NVIDIA GPU with 8 GB VRAM, CUDA 12.x driver | 16 GB+ VRAM (RTX 3080 / A10) |
| Docker CPU | 16 GB RAM | 32 GB RAM (slow — for testing only) |
| Bare-metal | Same as above | Same as above |

### Software (host machine)

| Tool | Version | Install |
|---|---|---|
| Docker Engine | ≥ 24 | [docs.docker.com/get-docker](https://docs.docker.com/get-docker/) |
| Docker Compose | ≥ 2.20 (Compose V2) | bundled with Docker Desktop |
| NVIDIA Container Toolkit | latest | [docs.nvidia.com/datacenter/cloud-native](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
| git | any | `apt install git` / `brew install git` |

> **Note for Windows users:** Use WSL 2 with the NVIDIA WSL driver. Docker Desktop with WSL 2 backend is fully supported.

Verify GPU access inside Docker before proceeding:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## 2. Repository Setup

```bash
# Clone the repository
git clone https://github.com/MichaelAkridge-NOAA/boxer.git
cd boxer

# Copy the environment template
cp .env.example .env
```

Open `.env` in your editor and fill in the required values:

```bash
# .env — required settings
METASHAPE_DIR=/absolute/path/to/your/metashape_export   # <-- CHANGE THIS
OUTPUT_DIR=./output          # where results are written
CKPT_DIR=./ckpts             # where model weights are cached
PROMPTS=coral                # SAM3 text prompts (comma-separated)
SAM3_MODE=image              # "image" or "video" (see §8)
```

> `METASHAPE_DIR` is the **only required** change. All other values have sensible defaults.

---

## 3. Metashape Export Format

The pipeline expects the following folder structure from Agisoft Metashape:

```
metashape_export/
├── cameras.xml          ← REQUIRED: camera intrinsics + extrinsics
├── images/              ← REQUIRED: per-frame RGB images (JPG or PNG)
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
├── dense.ply            ← RECOMMENDED: dense world-frame point cloud
└── depth_maps/          ← OPTIONAL: per-frame depth maps (EXR or TIFF)
    ├── frame_0001.exr
    └── ...
```

### How to export from Agisoft Metashape

1. **Align photos** and build a **Dense Cloud** in Metashape.
2. Export cameras:
   - `File → Export → Export Cameras…`
   - Format: **Agisoft XML** (`cameras.xml`)
   - ✅ Check **Export calibration** and **Export orientation**
3. Export dense cloud:
   - `File → Export → Export Point Cloud…`
   - Format: **Stanford PLY** (`dense.ply`)
   - Coordinate system: **local** (same as cameras.xml)
4. *(Optional)* Export depth maps:
   - `File → Export → Export Depth Maps…`
   - Format: **TIFF** or **EXR**, one file per camera

> **Tip — Ground alignment:** For best results run `Model → Align Chunk → Ground Control` in Metashape before exporting so the world Z-axis points up. If your scene uses a different convention, pass `--gravity_vec GX GY GZ` to the CLI (see §6).

---

## 4. Option A — Docker Compose (recommended)

### 4.1 GPU Deployment

**Step 1 — Build the image** (first run only, ~10 min):

```bash
docker compose build boxer
```

**Step 2 — Download model weights** (first run only, ~5 GB):

```bash
docker compose --profile setup run download_weights
```

This downloads SAM3 + BoxerNet weights from `https://huggingface.co/facebook/sam3` into `./ckpts/`.

**Step 3 — Run the full pipeline**:

```bash
docker compose up boxer
```

Watch the logs — the pipeline will print per-frame detection counts and an overall FPS estimate. When it finishes, results appear in `./output/<scene_name>/`.

**Step 4 — View results**:

```bash
ls ./output/
# boxer_sfm_3dbbs.csv           per-frame 3D OBBs
# boxer_sfm_3dbbs_fused.csv     globally fused instances
# sam3_masks/                   mask overlays (if --save_masks)
# coral_instances.ply           coloured point cloud
```

---

### 4.2 CPU-only Deployment

For machines without an NVIDIA GPU (testing / small datasets):

```bash
# Build the CPU image
docker compose -f docker-compose.yml -f docker-compose.cpu.yml build boxer

# Download weights (same step as GPU)
docker compose --profile setup run download_weights

# Run on CPU (significantly slower)
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up boxer
```

---

### Overriding pipeline settings without editing `.env`

Pass environment variables directly on the command line:

```bash
# Multi-class benthic survey
PROMPTS="coral,algae,rubble,sand" docker compose up boxer

# Temporal tracking mode (recommended for >70% image overlap)
SAM3_MODE=video docker compose up boxer

# Send outputs to a different folder
OUTPUT_DIR=/mnt/nas/results docker compose up boxer
```

---

## 5. Option B — Direct Python (no Docker)

### Install dependencies

```bash
# Create and activate a virtual environment (Python 3.12 recommended)
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Core Boxer dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Pipeline-specific dependencies
pip install huggingface_hub ultralytics scipy open3d tqdm opencv-python-headless

# Install Boxer itself (editable)
pip install -e .
```

### Download weights

```bash
bash scripts/download_weights.sh         # saves to ./ckpts/
# or point to a custom directory:
bash scripts/download_weights.sh /path/to/ckpts
```

### Run the pipeline

**Minimal run** (single prompt, default settings):

```bash
python run_boxer_sfm.py \
    --metashape_dir /path/to/metashape_export \
    --prompts "coral" \
    --fuse
```

**Full benthic survey with mask overlays and PLY export**:

```bash
python run_boxer_sfm.py \
    --metashape_dir /path/to/metashape_export \
    --prompts "coral,algae,rubble,sand" \
    --sam3_mode video \
    --fuse \
    --export_ply \
    --save_masks \
    --output_dir ./output
```

**Species-level coral survey**:

```bash
python run_boxer_sfm.py \
    --metashape_dir /path/to/metashape_export \
    --prompts "brain coral,staghorn coral,table coral,bleached coral" \
    --thresh2d 0.15 \
    --thresh3d 0.05 \
    --fuse \
    --export_ply
```

---

## 6. CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--metashape_dir` | *(required)* | Root folder of the Metashape export (must contain `cameras.xml`). |
| `--xml_filename` | `cameras.xml` | Name of the Agisoft cameras XML. |
| `--image_subdir` | *(auto-detected)* | Sub-folder containing images (`images/`, `raw_images/`, etc.). |
| `--ply_filename` | `dense.ply` | Name of the dense point-cloud PLY. |
| `--sdp_radius` | `5.0` | KD-tree radius in metres for per-frame point sampling. Increase for wide-baseline surveys. |
| `--gravity_vec GX GY GZ` | `0 0 -1` | Gravity direction in world space. Use `0 -1 0` for Y-up scenes. |
| `--skip_n` | `1` | Process every N-th frame (use `2` or `4` for fast previews). |
| `--start_n` | `0` | Zero-based index of the first frame to process. |
| `--max_n` | `99999` | Hard cap on number of frames. |
| `--prompts` | `coral` | Comma-separated SAM3 text prompts, e.g. `"coral,algae,rubble"`. |
| `--sam3_ckpt` | `./ckpts` | Directory containing SAM3 + BoxerNet weights. Auto-downloaded if absent. |
| `--sam3_mode` | `image` | `image` = independent per-frame; `video` = temporally tracked instance IDs. |
| `--thresh2d` | `0.1` | Minimum SAM3 confidence to keep a 2D detection (0–1). |
| `--ckpt` | *(auto-resolved)* | Explicit path to a BoxerNet `.ckpt` file. |
| `--thresh3d` | `0.1` | Minimum BoxerNet 3D confidence to keep a detection. Lower = more recall. |
| `--no_mask_filter` | off | Disable mask-filtered SDP — uses the full point cloud per detection. |
| `--fuse` | off | Run offline 3D box fusion across all frames after detection. |
| `--export_ply` | off | Write `coral_instances.ply` — a coloured point cloud, one colour per instance. |
| `--save_masks` | off | Save per-frame SAM3 mask overlay PNGs to `sam3_masks/`. |
| `--output_dir` | `./output` | Root directory for all output files. |
| `--write_name` | `boxer_sfm` | Prefix for CSV and other output files. |
| `--force_cpu` | off | Force CPU inference even if a GPU is available. |
| `--no_csv` | off | Skip writing detection CSV files. |

---

## 7. Understanding the Outputs

After a successful run the output directory contains:

```
output/
└── <scene_name>/
    ├── boxer_sfm_3dbbs.csv          # Per-frame 3D oriented bounding boxes
    ├── boxer_sfm_3dbbs_fused.csv    # Globally fused instances (with --fuse)
    ├── coral_instances.ply          # Coloured per-instance point cloud (with --export_ply)
    └── sam3_masks/                  # Per-frame PNG mask overlays (with --save_masks)
        ├── frame_00000.png
        ├── frame_00001.png
        └── ...
```

### CSV format (`boxer_sfm_3dbbs.csv`)

Each row is one 3D detection. Key columns:

| Column | Description |
|---|---|
| `time_ns` | Frame index (integer) |
| `cx, cy, cz` | OBB centre in world coordinates (metres) |
| `sx, sy, sz` | OBB half-extents (metres) |
| `yaw` | Rotation about the vertical axis (radians) |
| `prob` | BoxerNet 3D confidence (0–1) |
| `label` | SAM3 text prompt that produced this detection |

### Fused CSV (`boxer_sfm_3dbbs_fused.csv`)

Same columns as above but one row per globally unique instance, computed by IoU-based clustering across all frames.

### Coloured PLY (`coral_instances.ply`)

A standard point cloud where each detection's 3D support points are coloured by instance. Open with [CloudCompare](https://www.cloudcompare.org/) or [MeshLab](https://www.meshlab.net/) for visual inspection.

---

## 8. SAM3 Prompt Tuning

SAM3 uses open-vocabulary text prompts — no fine-tuning is required for common reef classes.

### Prompt examples

| Use case | `--prompts` value |
|---|---|
| General coral detection | `coral` |
| Targeted morphologies | `"brain coral,staghorn coral,table coral,bleached coral"` |
| Full benthic survey | `"coral,algae,rubble,sand,fish"` |
| Substrate only | `"rubble,sand,bare rock"` |

### Image mode vs video mode

| | `--sam3_mode image` | `--sam3_mode video` |
|---|---|---|
| Speed | Faster | Slower (pre-loads all frames) |
| Instance IDs | Independent per frame | Temporally consistent across frames |
| Best for | Spot checks, large frame counts | High-overlap surveys (>70%), 3D tracking |

### Confidence thresholds

- `--thresh2d 0.1` — Lowers the bar for SAM3 to produce a mask. Increase to `0.3–0.5` if you get too many false positives on sand / water column.
- `--thresh3d 0.05–0.1` — BoxerNet was trained on indoor objects so coral OBBs score lower than chairs and tables. Keep this low (0.05–0.15) for coral surveys.

---

## 9. Troubleshooting

### `METASHAPE_DIR` is not set

```
Error: Set METASHAPE_DIR in .env
```

Make sure you have edited `.env` and that `METASHAPE_DIR` points to an existing directory containing `cameras.xml`.

### No cameras found after parsing `cameras.xml`

The loader auto-searches for images in common sub-folders (`images/`, `raw_images/`, `frames/`, `photos/`). If your images live elsewhere, pass:

```bash
--image_subdir my_custom_subfolder
```

### "Could not find a BoxerNet checkpoint"

```
FileNotFoundError: Could not find a BoxerNet checkpoint.
```

Run the weight downloader and confirm the `./ckpts/` directory is non-empty:

```bash
bash scripts/download_weights.sh
ls ckpts/
```

If running in Docker, ensure the `ckpts` volume is mounted correctly (`-v ./ckpts:/app/ckpts`).

### SAM3 model fails to load (Ultralytics not installed)

The detector will fall back to the native `sam3` Python package. Install one of:

```bash
pip install ultralytics          # recommended
# or
pip install sam3                 # native facebook/sam3 package
```

### GPU not detected inside Docker

```
==> Device: cpu   (expected: cuda)
```

1. Verify the host driver: `nvidia-smi`
2. Verify NVIDIA Container Toolkit: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
3. Ensure `docker-compose.yml` is used (not `docker-compose.cpu.yml`) and the `deploy.resources.reservations` block is present.

### Dense PLY takes too long / runs out of memory

Reduce the KD-tree query radius (fewer points sampled per frame):

```bash
--sdp_radius 2.0     # default is 5.0 metres
```

For very large surveys (>500 M points), consider down-sampling the PLY in CloudCompare before running the pipeline.

### Wrong gravity direction (boxes tilted or upside-down)

If the Metashape project uses Y-up coordinates, pass:

```bash
--gravity_vec 0 -1 0
```

The gravity vector points in the **direction of gravitational pull** in world space (e.g. `[0, 0, -1]` means the ground is at lower Z values).
