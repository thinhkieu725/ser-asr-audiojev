#!/usr/bin/env bash
# PLAN 0.3: environment on the Linux server. Adjust the CUDA wheel index to the driver.
set -euo pipefail
conda create -n ser-audiojev python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate ser-audiojev
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn --no-build-isolation || echo "flash-attn did not build; sdpa will be used"
pip install -e .
