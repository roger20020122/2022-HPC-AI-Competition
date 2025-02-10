#!/bin/bash
#PBS -N test
#PBS -P rw67
#PBS -r y
#PBS -q gpuvolta  
#PBS -l walltime=1:30:00 
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=100GB
#PBS -e /scratch/rw67/pc0922/error14a.txt
#PBS -o /scratch/rw67/pc0922/output14a.txt

cd /scratch/rw67/pc0922/dna_torch
module load python3/3.9.2 cuda/11.4.1 cudnn/8.2.2-cuda11.4 nccl/2.10.3-cuda11.4 openmpi/4.1.2

module prepend-path LD_PRELOAD libmpi.so
source /home/552/pc0922/miniconda3/etc/profile.d/conda.sh
conda init bash >/dev/null
. ~/.bashrc
conda activate dna2

python -u deepLearningBasedDNASequenceFastDecoding/optim/optim.py --log_dir logs/log --data_dir deepLearningBasedDNASequenceFastDecoding/data/preprocessed_CTCF_fimo_data --exp_name optim --val_freq 5 --save_freq 10000 --batch_size 128 --dist False

