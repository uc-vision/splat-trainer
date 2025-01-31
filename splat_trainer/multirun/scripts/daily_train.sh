#!/bin/bash

export PIXI_CACHE_DIR=/local/pixi
DATE=$(date +"%Y-%m-%d__%H-%M-%S")
PROJECT="splat_trainer_daily_multirun"
BASE_PATH=/local/${PROJECT}

mkdir -p ${BASE_PATH}/${DATE}
LOG_FILE="${BASE_PATH}/${DATE}/daily_multirun_log_${DATE}.log"

echo "Script started at $(date)" >> "$LOG_FILE"
source ~/.bashrc

cd /local/repo/splat-workspace
git config -f .gitmodules submodule.splat-trainer.branch multirun_test
git submodule update --remote


export TORCH_EXTENSIONS_DIR=$HOME/.cache/torch_extensions/py311_cu121_daily_multirun
rm -rf $HOME/.cache/torch_extensions/py311_cu121_daily_multirun


pixi run splat-trainer-multirun +multirun=daily_multirun logger.group=${DATE} >> "$LOG_FILE" 2>&1

echo "Script completed at $(date)" >> "$LOG_FILE"