#!/bin/bash
# Depth / multi-control training launcher for PixelDiT.
#
# Usage:
#   bash t2i/train_control.sh t2i/configs_t2i/pixeldit_depth_control_v1.yaml
#   bash t2i/train_control.sh t2i/configs_t2i/pixeldit_multicontrol_v1.yaml [...extra args...]
#
# Override np / work_dir on the fly:
#   NP=2 WORK_DIR=./my_runs/depth_v1 bash t2i/train_control.sh \
#       t2i/configs_t2i/pixeldit_depth_control_v1.yaml

set -e

work_dir=${WORK_DIR:-}
np=${NP:-2}
master_port=${MASTER_PORT:-29502}
name=${NAME:-pixeldit_control_run}

if [[ $1 == *.yaml ]]; then
    config=$1
    shift
else
    echo "Usage: bash t2i/train_control.sh <path/to/config.yaml> [extra pyrallis flags]"
    exit 1
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

case "$config" in
    /*) config_path="$config" ;;
    t2i/*) config_path="$repo_root/$config" ;;
    *) config_path="$script_dir/$config" ;;
esac

cd "$script_dir"

torchrun --nproc_per_node=$np --master_port=$master_port \
    train_control.py \
    --config_path=$config_path \
    --name=$name \
    ${work_dir:+--work_dir=$work_dir} \
    "$@"
