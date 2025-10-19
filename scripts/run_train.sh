#!/bin/bash
# This script runs the main training pipeline.

echo "Starting model training..."

python -m src.train --config configs/base.yaml

echo "Training finished."
