#!/bin/bash
#SBATCH --job-name=ebird_1st
#SBATCH --output=job_output_test.txt
#SBATCH --error=job_error_test.txt
#SBATCH --ntasks=1
#SBATCH --time=10:59:00
#SBATCH --mem-per-cpu=50Gb
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

#!/bin/bash # The interpreter used to execute the script
# Lines beginning with #SBATCH specify your computing resources and other logistics about how to run your job.
#Access computing resources allocated to the MLRE course account,the section may be 006 or 007 depending on the student.
SBATCH --account=eecs498f25s007_class
# Specify the maximum runtime (in Hours:Minutes:Seconds). If your job hits that runtime, it will be terminated.
SBATCH --time=5:00:00
# Specify whether to use a GPU (partition=gpu) or CPU (partition=standard). If you use the GPU partition, only request one
#GPU (gpus=1). Note that if you use the standard partition, you may
#have to remove the gpu configuration.
SBATCH --partition=gpu
SBATCH --gpus=1
# There are also more specific settings for configuring your CPU/GPU. You can reference the documentation for more information.
# Specify the amount of memory you need. If your job exceeds this #memory limit, it will be terminated.
SBATCH --mem=32g
# Name this job and the output file
SBATCH --job-name=resnet18_test_kenya
#SBATCH --output=example_output.out
# Receive an email when your job starts and ends
SBATCH --mail-user=dshayla@umich.edu
SBATCH --mail-type=BEGIN,END
# Load Python and any other desired modules module load python/3.12.1
# Specify the script you want to run

# module load miniconda/3
# conda activate eco
# export COMET_API_KEY=$COMET_API_KEY
# export HYDRA_FULL_ERROR=1
python train.py  ++auto_lr_find="False" args.config=configs/SatBird-Kenya/resnet18.yaml args.run_id=1
