#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Editable launch parameters. Evaluation options and paths live in the YAML file.
config_path=configs/eval/llada/gsm8k/justgrpo-llada-gsm8k.yaml
num_machines=1
processes_per_machine=8
machine_rank=0
main_process_ip=127.0.0.1
main_process_port=29500

export TOKENIZERS_PARALLELISM=false

exec torchrun \
  --nnodes "$num_machines" \
  --nproc_per_node "$processes_per_machine" \
  --node_rank "$machine_rank" \
  --master_addr "$main_process_ip" \
  --master_port "$main_process_port" \
  eval.py --config "$config_path"
