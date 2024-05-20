#!/bin/bash

#salloc --account open --job-name "InteractiveJob" --cpus-per-task 4 --mem 100G --time 8:00:00

srun --nodes=1 --ntasks=4 --gres=gpu:1 --time=8:00:00 --mem=40GB --partition=mahony --job-name=terminal --pty /bin/bash
