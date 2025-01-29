#!/bin/bash
#SBATCH -o /home/%u/slogs/sl_%A.out
#SBATCH -e /home/%u/slogs/sl_%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1  # use 1 GPU
#SBATCH --mem=14000  # memory in Mb
#SBATCH --partition=ILCC_GPU
#SBATCH -t 12:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=2

set -e # fail fast

dt=$(date '+%d_%m_%y_%H_%M');
echo "I am job ${SLURM_JOB_ID}"
echo "I'm running on ${SLURM_JOB_NODELIST}"
echo "Job started at ${dt}"

# ====================
# Activate Anaconda environment
# ====================
source /home/s2155899/miniconda3/bin/activate mlp

# ====================
# RSYNC data from /home/ to /disk/scratch/
# ====================
export SCRATCH_HOME=/disk/scratch/s2155899/GeoGuessing
export DATA_HOME=/home/GeoGuessing
export DATA_SCRATCH=${SCRATCH_HOME}/
mkdir -p ${SCRATCH_HOME}/pgr/data
rsync --archive --update --compress --progress ${DATA_HOME}/ ${DATA_SCRATCH}

# ====================
# Run training. Here we use src/gpu.py
# ====================
echo "Creating directory to save model predictions"
export OUTPUT_DIR=${SCRATCH_HOME}/predictions
mkdir -p ${OUTPUT_DIR}

# This script does not actually do very much. But it does demonstrate the principles of training
# python src/gpu.py \
# 	--data_path=${DATA_SCRATCH}/train.txt \
# 	--output_dir=${OUTPUT_DIR}

# ====================
# Run prediction. We will save outputs and weights to the same location but this is not necessary
# ====================
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
python aya_on_anli.py \
	--data_path=${DATA_SCRATCH}/ \
	--model_dir=${OUTPUT_DIR} 

# ====================
# RSYNC data from /disk/scratch/ to /home/. This moves everything we want back onto the distributed file system
# ====================
OUTPUT_HOME=/home/s2148449/ayaonanli
mkdir -p ${OUTPUT_HOME}
rsync --archive --update --compress --progress ${OUTPUT_DIR} ${OUTPUT_HOME}

# ====================
# Finally we cleanup after ourselves by deleting what we created on /disk/scratch/
# ====================
rm -rf ${OUTPUT_DIR}


echo "Job ${SLURM_JOB_ID} is done!"
