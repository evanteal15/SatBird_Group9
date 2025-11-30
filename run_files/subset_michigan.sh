#!/bin/bash
#SBATCH --account=eecs498f25s006_class
#SBATCH --job-name=subset_michigan          # Job name
#SBATCH --output=logs/%x_%j.out             # Log output
#SBATCH --error=logs/%x_%j.err              # Error logs
#SBATCH --time=01:00:00                     # Should be plenty for subsetting
#SBATCH --partition=standard                # CPU-only partition
#SBATCH --cpus-per-task=4                   # 4 CPU cores
#SBATCH --mem=32G                           # 32 GB memory
#SBATCH --mail-user=alexdev@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# --- Load and activate environment ---
module purge
source ~/.bashrc

# Move to the directory where you ran sbatch
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Use same Python environment as training job
PYTHON_BIN="$(pwd)/satbird_env/bin/python"
echo "Using Python binary at: $PYTHON_BIN"

# --- Run subsetting script ---
echo "Starting Michigan subset on $(hostname) at $(date)"
$PYTHON_BIN subset_michigan.py
echo "Michigan subset finished at $(date)"
