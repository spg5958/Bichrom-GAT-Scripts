#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:2
#SBATCH --mem=100GB
#SBATCH --partition=mahony
#SBATCH --job-name=train_bichrom
#SBATCH --output=../example_outpu/slurm-%x.out
umask 007


# Slurm script to submit job for training seq-net and GAT-net/Bimodal-net.
# Modify this as per your cluster specification

source ~/conda_init.sh
conda activate pytorch_bichrom

SECONDS=0

python train_bichrom.py

ELAPSED="Elapsed(trainNN.sh): $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo ""
echo $ELAPSED
