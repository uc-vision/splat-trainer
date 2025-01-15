#!/bin/bash


DATE=$(date +"%Y-%m-%d__%H-%M-%S")
LOG_FILE="/local/splat_trainer_daily_multirun/daily_train_log_${DATE}.log"

echo "Script started at $(date)" >> "$LOG_FILE"

mkdir /local/splat_trainer_daily_multirun
cd /local/splat_trainer_daily_multirun
rm -rf /local/splat_trainer_daily_multirun/splat-trainer
git clone git@github.com:uc-vision/splat-trainer.git

cd /local/splat_trainer_daily_multirun/splat-trainer
git clone git@github.com:uc-vision/taichi-splatting.git

cd /local/splat_trainer_daily_multirun/splat-trainer/taichi-splatting
pip install -e .

cd /local/splat_trainer_daily_multirun/splat-trainer
pip install -e .

source ~/.bashrc
conda activate splat-trainer

grid-search-trainer \
    logger=wandb \
    logger.group=${DATE} \
    logger.entity=UCVision \
    scene=sh \
    +project=splat_trainer_daily_multirun  \
    hydra.sweep.dir=/local/splat_trainer_daily_multirun/${DATE} \
    trainer.num_logged_images=-1 \
    trainer.log_worst_images=-1 \
    >> "$LOG_FILE" 2>&1

echo "Script completed at $(date)" >> "$LOG_FILE"