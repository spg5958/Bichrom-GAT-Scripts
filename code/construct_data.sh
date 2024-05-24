#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=100GB
#SBATCH --partition=mahony
#SBATCH --job-name=construct_data
#SBATCH --output=../../example_output/slurm-%x.out
umask 007


# Slurm script to submit job for constructing training and test data.
# Modify this as per your cluster specification

source ~/conda_init.sh
conda activate pytorch_bichrom

SECONDS=0

#TMPDIR=/home/spg5958/tmp
#echo $TMPDIR

python construct_data.py

ELAPSED="Elapsed(construct_data.sh): $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo ""
echo $ELAPSED
echo ""
