#!/bin/sh
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1  # 
#SBATCH --gres=gpu:1
wandb agent i4ds_radio_sunburst_detection/radio_sunburst_detection/rlavlwvy