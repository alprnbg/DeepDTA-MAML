#!/bin/sh
set -e
export GPU_ID=0

echo $GPU_ID

cd ..
export DATASET_DIR="datasets/"
export CUDA_VISIBLE_DEVICES=$GPU_ID
# Activate the relevant virtual environment:


python train_maml_system.py --name_of_args_json_file experiment_config/config.json

python train_maml_system.py --name_of_args_json_file experiment_config/config2.json