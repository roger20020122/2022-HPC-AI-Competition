#!/bin/bash
#PBS -N Leopard_NCI
#PBS -P rw67
#PBS -r y
#PBS -q gpuvolta  
#PBS -l storage=gdata/ik06
#PBS -l walltime=01:30:00 
#PBS -l ncpus=384
#PBS -l ngpus=32
#PBS -l mem=2000GB
#PBS -M F84091087@gs.ncku.edu.tw
#PBS -m e


###########################
export TF_ENABLE_ONEDNN_OPTS=0

# setup conda environment 
# -- change the path to your own conda directory
source /home/552/lh9988/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate leopard

#load modules for gpu support
module load cuda
module load cudnn
module load nccl
module load openmpi

# run the bechmark over one GPUs
# -- change the path to your own 
source ./multinodes_train.sh


