#!/bin/bash
# This script runs the model evaluation pipeline.

echo "Starting model evaluation..."

python -m src.evaluate --config configs/base.yaml

echo "Evaluation finished."
