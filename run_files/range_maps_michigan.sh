#!/bin/bash
#SBATCH --account=eecs498f25s006_class
#SBATCH --job-name=range_maps_mi
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-user=alexdev@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# --- Environment setup ---
module purge
source ~/.bashrc

PYTHON_BIN="$(pwd)/satbird_env/bin/python"
echo "Using Python binary at: $PYTHON_BIN"

echo "Starting Michigan range map generation on $(hostname) at $(date)"

# --- Run Michigan range map generation ---
echo "Generating range maps for Michigan subset"
$PYTHON_BIN ./data_processing/ebird/get_range_maps_michigan.py

echo "Michigan range map generation finished at $(date)"
