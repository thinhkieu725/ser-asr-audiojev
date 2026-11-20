# Source before every run:  source scripts/env.sh
# PLACEHOLDERS: set to the server's GPU id and a large disk.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME=${HF_HOME:-/path/to/large/disk/hf_home}
export TOKENIZERS_PARALLELISM=false
