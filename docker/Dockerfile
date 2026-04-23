# docker/Dockerfile
# Boxer SfM pipeline – GPU image (CUDA 12.4 + Python 3.12)
#
# Build:
#   docker build -f docker/Dockerfile -t boxer-sfm:gpu .
#
# Run:
#   docker run --gpus all \
#     -v /path/to/metashape_export:/data/input:ro \
#     -v ./output:/data/output \
#     -v ./ckpts:/app/ckpts \
#     -e PROMPTS="coral" \
#     boxer-sfm:gpu

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# ── System dependencies ──────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-distutils \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python python3.12 1

# ── Upgrade pip ──────────────────────────────────────────────────────────────
RUN python -m pip install --no-cache-dir --upgrade pip

# ── PyTorch (CUDA 12.4) ──────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ── Python dependencies ──────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    huggingface_hub \
    ultralytics \
    open3d \
    scipy \
    tqdm \
    opencv-python-headless \
    numpy \
    Pillow

# ── Copy source & install ────────────────────────────────────────────────────
WORKDIR /app
COPY . /app

# Install Boxer in editable mode (pyproject.toml must exist)
RUN pip install --no-cache-dir -e . 2>/dev/null || true

# ── Runtime environment ──────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV PROMPTS=coral
ENV SAM3_MODE=image

# Create data mount-points
RUN mkdir -p /data/input /data/output /app/ckpts

ENTRYPOINT ["/app/docker/entrypoint.sh"]
