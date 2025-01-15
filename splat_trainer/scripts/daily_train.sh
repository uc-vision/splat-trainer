#!/bin/bash


DATE=$(date +"%Y-%m-%d__%H-%M-%S")
BASE_PATH=/local/splat_trainer_daily_multirun

mkdir ${BASE_PATH}/${DATE}
LOG_FILE="${BASE_PATH}/${DATE}/daily_train_log_${DATE}.log"

echo "Script started at $(date)" >> "$LOG_FILE"

cd ${BASE_PATH}
rm -rf ${BASE_PATH}/splat-trainer
git clone -b daily-multirun git@github.com:uc-vision/splat-trainer.git

cd ${BASE_PATH}/splat-trainer
git clone git@github.com:uc-vision/taichi-splatting.git

cd ${BASE_PATH}/splat-trainer/taichi-splatting
pip install -e .

cd ${BASE_PATH}/splat-trainer
pip install -e .

source ~/.bashrc
conda activate splat-trainer

grid-search-trainer \
    logger=wandb \
    logger.group=${DATE} \
    logger.entity=UCVision \
    scene=sh \
    +project=splat_trainer_daily_multirun  \
    hydra.sweep.dir=${BASE_PATH}/${DATE} \
    trainer.num_logged_images=-1 \
    trainer.log_worst_images=-1 \
    >> "$LOG_FILE" 2>&1

echo "Script completed at $(date)" >> "$LOG_FILE"