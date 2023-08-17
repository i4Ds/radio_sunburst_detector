#!/bin/sh
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=top6
#SBATCH --exclude=gpu23a,gpu23b,gpu23c,gpu23d
##SBATCH --nodelist=gpu23a
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH
export PATH="${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
wandb agent i4ds_radio_sunburst_detection/radio_sunburst_detection/pg4rf2ys