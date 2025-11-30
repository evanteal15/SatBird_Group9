#!/bin/bash
#SBATCH --account=eecs498f25s006_class
#SBATCH --job-name=satbird_train         # Job name
#SBATCH --output=logs/%x_%j.out          # Save logs in logs/ directory (jobname_jobid.out)
#SBATCH --error=logs/%x_%j.err           # Error logs
#SBATCH --time=06:00:00                  # Max 6 hours
#SBATCH --partition=gpu                  # Use GPU partition
#SBATCH --gpus=1                         # Request 1 GPU
#SBATCH --cpus-per-task=4                # 4 CPU cores
#SBATCH --mem=32G                        # 32 GB memory
#SBATCH --mail-user=alexdev@umich.edu,kevx@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# --- Load and activate environment ---
module purge
source ~/.bashrc

# Use absolute path to python from local satbird_env
PYTHON_BIN="$(pwd)/satbird_env/bin/python"
echo "Using Python binary at: $PYTHON_BIN"

# --- Run training ---
echo "Starting training on $(hostname) at $(date)"
echo "Config file: $1"

# --- Set TORCH_HOME to local directory to avoid permission issues for torchvision.models ---
export TORCH_HOME=$(pwd)/.torch
$PYTHON_BIN train.py args.config=$1 ++auto_lr_find="False" hydra.run.dir=.

echo "Training finished at $(date)"

