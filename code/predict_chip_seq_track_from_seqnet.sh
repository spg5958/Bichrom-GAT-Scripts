#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --partition=mahony
#SBATCH --job-name=predict_chip_seq_track_from_seqnet
#SBATCH --output=../output/slurm-%x.out
umask 007


# Slurm script to submit job for constructing training and test data.
# Modify this as per your cluster specification

source ~/conda_init.sh
conda activate pytorch_bichrom

SECONDS=0

python predict_chip_seq_track_from_seqnet.py

ELAPSED="Elapsed(trainNN.sh): $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo ""
echo $ELAPSED
