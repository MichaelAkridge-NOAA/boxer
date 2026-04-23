#!/usr/bin/env bash
# scripts/download_weights.sh
#
# Download SAM3 + Boxer weights from HuggingFace.
# Both model families are hosted at: https://huggingface.co/facebook/sam3
#
# Usage:
#   bash scripts/download_weights.sh [TARGET_DIR]
#
# If TARGET_DIR is not specified, weights are saved to ./ckpts/

set -euo pipefail

HF_REPO="facebook/sam3"
TARGET_DIR="${1:-$(dirname "$(cd "$(dirname "$0")" && pwd)")/ckpts}"

mkdir -p "$TARGET_DIR"
echo "==> Downloading SAM3 + Boxer weights from HuggingFace (${HF_REPO})"
echo "==> Target directory: ${TARGET_DIR}"

# ── Prefer huggingface-cli if available ─────────────────────────────────────
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download \
        "$HF_REPO" \
        --local-dir "$TARGET_DIR" \
        --exclude "*.md" "*.txt" "*.gitattributes"
    echo "==> Download complete via huggingface-cli."
    exit 0
fi

# ── Fallback: Python huggingface_hub ────────────────────────────────────────
python3 - <<PYEOF
import sys
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("ERROR: huggingface_hub is not installed.")
    print("       Run:  pip install huggingface_hub")
    sys.exit(1)

print("Downloading via huggingface_hub.snapshot_download …")
local = snapshot_download(
    repo_id="$HF_REPO",
    local_dir="$TARGET_DIR",
    ignore_patterns=["*.md", "*.txt", ".gitattributes"],
)
print(f"Downloaded to: {local}")
PYEOF

echo "==> Download complete."
