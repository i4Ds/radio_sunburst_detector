# Radio Sunburst Detector
Results: https://wandb.ai/i4ds_radio_sunburst_detection/radio_sunburst_detection/reports/Radio-Sunburst-Detector--Vmlldzo1NTkwNDQy
## Core Scripts

- `main.py`: Main script to execute the entire pipeline. Updated parameters.
- `modelbuilder.py`: Contains code for building and compiling models. Fixed transfer learning issues.
- `train_utils.py`: Utilities for training models. Formatted with Black.

## Data Preparation and Generation

- `data_generation.ipynb`: Notebook with scripts for data generation. Added background subtraction.
- `create_background_images.ipynb`: Scripts to generate background images.
- `configure_dataframes.py`: Manages DataFrame configurations. Cleaned up and formatted.
- `extract_instrument_data.py`: Scripts for specific instrument data extraction. 
- `burst_data_generation.py`: Scripts for generating burst-specific data.
- `non_burst_data_generation.py`: Scripts for generating non-burst data.

## Data Inspection and Validation

- `check_data.ipynb`: Notebook to check data statistics.
- `check_labels.ipynb`: Notebook to validate label information. 
- `check_labels.py`: Script version for checking labels. 

## Configuration and Parameters

- `config.yaml`: YAML configuration file. Introduced wandb support.
- `sweep.yaml`: YAML file for wandb sweep configurations.
- `sweep_no_folds.yaml`: Alternate sweep configuration without folds.

## Metrics and Analysis

- `metric_utils.py`: Utility functions for metrics. Added batch size parameter.
- `model_analysis.ipynb`: Notebook for model analysis using tf-explain.

## Misc

- `requirements.txt`: List of Python dependencies.
- `sbatch.sh`: Batch job script.
- `sbatch_main.sh`: Main batch job script.
- `.gitignore`: Ignored files and directories.
- `README.md`: Repository readme file. Added hello.

## Data Files

- `burst_list.xlsx`: Excel sheet containing list of bursts.
- `data.xlsx`: Prepared data in Excel format.

## Images

- `original_image.png`: Sample original image.
- `processed_image.png`: Sample processed image.

## Legacy/Deprecated

- `model_base_configs`: Older model configurations.
