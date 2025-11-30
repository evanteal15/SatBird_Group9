#!/bin/bash
#SBATCH --account=eecs498f25s006_class
#SBATCH --job-name=satbird_test         # Job name
#SBATCH --output=logs/%x_%j.out          # Save logs in logs/ directory (jobname_jobid.out)
#SBATCH --error=logs/%x_%j.err           # Error logs
#SBATCH --time=01:00:00                  # Max 1 hours
#SBATCH --partition=gpu                  # Use GPU partition
#SBATCH --gpus=1                         # Request 1 GPU
#SBATCH --cpus-per-task=4                # 4 CPU cores
#SBATCH --mem=32G                        # 32 GB memory
#SBATCH --mail-user=alexdev@umich.edu,kevx@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# --- Load and activate environment ---
module purge
source ~/.bashrc
conda activate satbird

# --- Run training ---
echo "Starting testing on $(hostname) at $(date)"
echo "Config file: $1"

python test.py args.config=$1 ++auto_lr_find="False" hydra.run.dir=.

echo "Testing finished at $(date)"

