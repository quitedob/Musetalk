#!/bin/bash

# MuseTalk Training Script
# This script combines both training stages for the MuseTalk model
# Usage: sh train.sh [stage1|stage2|stage1_v2|stage2_v2]
# Example: sh train.sh stage1  # To run stage 1 training
# Example: sh train.sh stage2  # To run stage 2 training
# Example: sh train.sh stage1_v2  # To run v2 stage 1 training
# Example: sh train.sh stage2_v2  # To run v2 stage 2 training

# Check if stage argument is provided
if [ $# -ne 1 ]; then
    echo "Error: Please specify the training stage"
    echo "Usage: ./train.sh [stage1|stage2|stage1_v2|stage2_v2]"
    exit 1
fi

STAGE=$1

# Validate stage argument
case "$STAGE" in
    stage1|stage2|stage1_v2|stage2_v2)
        ;;
    *)
        echo "Error: Invalid stage. Must be one of 'stage1', 'stage2', 'stage1_v2', 'stage2_v2'"
        exit 1
        ;;
esac

# Launch distributed training using accelerate
# --config_file: Path to the GPU configuration file
# --main_process_port: Port number for the main process, used for distributed training communication
# train.py: Training script
# --config: Path to the training configuration file
echo "Starting $STAGE training..."
accelerate launch --config_file ./configs/training/gpu.yaml \
                  --main_process_port 29502 \
                  train.py --config ./configs/training/$STAGE.yaml

echo "Training completed for $STAGE" 
