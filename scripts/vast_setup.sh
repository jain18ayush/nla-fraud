#!/usr/bin/env bash
# Bootstrap a fresh Vast.ai box for Phase 4 (SFT).
# Usage, on the box:  bash scripts/vast_setup.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# HF_HOME must be set before any snapshot_download, and must persist into
# tmux sessions started later -- hence .bashrc as well as this shell.
export HF_HOME="${HF_HOME:-$REPO_DIR/.hf}"
grep -q 'export HF_HOME=' ~/.bashrc 2>/dev/null || \
    echo "export HF_HOME=$HF_HOME" >> ~/.bashrc

echo "=== disk ==="
df -h / | tail -1
avail_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$avail_gb" -lt 55 ]; then
    echo "WARNING: only ${avail_gb}G free. Two 7B bf16 checkpoints (~30G) plus"
    echo "the torch venv (~10G) need ~45G. Expect to run out."
fi

echo "=== venv ==="
uv sync

echo "=== weights ==="
# The training scripts call snapshot_download themselves, but doing it up front
# means a disk-space failure surfaces now rather than 40 minutes into a run.
uv run python - <<'PY'
import os, yaml
from huggingface_hub import snapshot_download
cfg = yaml.safe_load(open("configs/experiment.yaml"))["sft"]
for key in ("ar_model_id", "av_model_id"):
    mid = cfg[key]
    print(f"--- {mid}", flush=True)
    snapshot_download(mid)
print("HF_HOME =", os.environ.get("HF_HOME"))
PY

echo "=== check ==="
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
test -f data/summaries.parquet \
    && echo "summaries.parquet OK" \
    || echo "MISSING data/summaries.parquet -- rsync it from your laptop"

echo "=== disk after ==="
df -h / | tail -1
echo "Setup complete."
