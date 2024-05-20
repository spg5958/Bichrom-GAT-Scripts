#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:2
#SBATCH --mem=100GB
#SBATCH --partition=mahony
#SBATCH --job-name=train_bichrom
#SBATCH --output=../output/slurm-%x.out
umask 007

source ~/conda_init.sh
conda activate pytorch_bichrom

#export CUDA_VISIBLE_DEVICES=2,3

SECONDS=0

TMPDIR=/home/spg5958/tmp
echo $TMPDIR

python train_bichrom.py

ELAPSED="Elapsed(trainNN.sh): $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo ""
echo $ELAPSED
