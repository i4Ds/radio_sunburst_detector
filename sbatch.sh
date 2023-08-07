#!/bin/sh
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=1  # 
#SBATCH --gres=gpu:1
#SBATCH --partition=top6
conda activate radio_sunburst_detection
wandb agent i4ds_radio_sunburst_detection/radio_sunburst_detection/0c0it7ps