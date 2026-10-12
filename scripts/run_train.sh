#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Editable launch parameters. Training options and paths live in the YAML file.
config_path=configs/train/llada/gsm8k/repara-llada-gsm8k-fix2.yaml
# LLaDA: configs/runtime/fsdp.yaml; SDAR: configs/runtime/fsdp_sdar.yaml; DDP: configs/runtime/ddp.yaml.
accelerate_config_path=configs/runtime/fsdp.yaml
num_machines=1
processes_per_machine=8
machine_rank=0
main_process_ip=127.0.0.1
main_process_port=29500
num_processes=$((num_machines * processes_per_machine))

export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLASH_ATTENTION_DETERMINISTIC=1
export NNODES="$num_machines"
export NPROC_PER_NODE="$processes_per_machine"
export CONFIG_FILE="$accelerate_config_path"

exec accelerate launch \
  --config_file "$accelerate_config_path" \
  --num_processes "$num_processes" \
  --num_machines "$num_machines" \
  --machine_rank "$machine_rank" \
  --main_process_ip "$main_process_ip" \
  --main_process_port "$main_process_port" \
  train.py --config "$config_path"
