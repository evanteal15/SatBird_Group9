#!/bin/bash

#SBATCH --account=eecs498f25s006_class

#SBATCH --job-name=ebird_data_step1

# Specify the maximum runtime (in Hours:Minutes:Seconds). If your job hits that runtime, it will be terminated.
#SBATCH --time=1:00:00

# Specify whether to use a GPU (partition=gpu) or CPU (partition=standard). If you use the GPU partition, only request one GPU (gpus=1)
#SBATCH --partition=standard

# There are also more specific settings for configuring your CPU/GPU. You can reference the documentation for more information.
# Specify the amount of memory you need. If your job exceeds this memory limit, it will be terminated.
#SBATCH --mem=32g

# Receive an email when your job starts and ends
#SBATCH --mail-user=evanteal@umich.edu
#SBATCH --mail-type=BEGIN,END

#SBATCH --output=job_output_test.txt


module load python/3.12.1
# Specify the script you want to run
python filter2.py EBD/ebd_US-MI_smp_relSep-2025.txt
