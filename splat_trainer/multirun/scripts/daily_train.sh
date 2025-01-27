#!/bin/bash


DATE=$(date +"%Y-%m-%d__%H-%M-%S")
PROJECT="splat_trainer_daily_multirun"
BASE_PATH=/local/${PROJECT}

mkdir -p ${BASE_PATH}/${DATE}
LOG_FILE="${BASE_PATH}/${DATE}/daily_multirun_log_${DATE}.log"

echo "Script started at $(date)" >> "$LOG_FILE"

cd ${BASE_PATH}
rm -rf ${BASE_PATH}/splat-trainer
git clone --recursive -b multirun_test git@github.com:uc-vision/splat-trainer.git


cd ${BASE_PATH}/splat-trainer/taichi-splatting
pip install -e .

cd ${BASE_PATH}/splat-trainer
pip install -e .

source ~/.bashrc
conda activate splat-trainer

rm -rf $HOME/.cache/torch_extensions/py311_cu121_room347

splat-trainer-multirun +multirun=daily_multirun logger.group=${DATE} >> "$LOG_FILE" 2>&1

echo "Script completed at $(date)" >> "$LOG_FILE"

rm -rf ${BASE_PATH}/splat-trainer