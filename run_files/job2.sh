#!/bin/bash
#SBATCH --account=eecs498f25s006_class
#SBATCH --time=5:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=ebird_1st
#SBATCH --output=job_output_test.out
#SBATCH --mail-user=dshayla@umich.edu
#SBATCH --mail-type=BEGIN,END

# Load Python or Conda environment
module load python/3.10

# --- Setup Virtual Environment ---
VENV_DIR=$HOME/venvs/ebird_venv

# Create the venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip and install all requirements
python3 -m pip install --upgrade "pip<24.1"
python3 -m pip install numpy==1.24.1 pandas==1.1.5 --prefer-binary
pip install --prefer-binary -r /scratch/eecs498f25s006_class_root/eecs498f25s006_class/shared_data/SatBird_Group9/requirements/requirements.txt

export HYDRA_FULL_ERROR=1

TRAIN_PY="/scratch/eecs498f25s006_class_root/eecs498f25s006_class/shared_data/SatBird_Group9/train.py"

# run_id is for reproducibility; dev=True to run dev training (set to False for full training)
python3 "$TRAIN_PY" \
    --config-path /scratch/eecs498f25s006_class_root/eecs498f25s006_class/shared_data/SatBird_Group9/configs \
    --config-name SatBird-Kenya/resnet18.yaml \
    ++args.base_dir="/scratch/eecs498f25s006_class_root/eecs498f25s006_class/shared_data/SatBird_Group9" \
    ++args.run_id=1 \
    ++args.dev=True \
    +args.config="configs/SatBird-Kenya/resnet18.yaml" \
    ++args.train_csv="/scratch/eecs498f25s006_class_root/eecs498f25s006_class/shared_data/SatBird_Group9/new_kenya2/train_split.csv" \
    ++args.val_csv="/scratch/eecs498f25s006_class_root/eecs498f25s006_class/shared_data/SatBird_Group9/new_kenya2/val_split.csv"
    
