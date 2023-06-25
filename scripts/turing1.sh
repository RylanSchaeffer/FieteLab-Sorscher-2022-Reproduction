#!/bin/bash

id=${1}
gpu_id=${2}

source /afs/cs.stanford.edu/u/rschaef/.bashrc.user

# Activate conda environment.
conda activate sorscher_reproduction

# Change to the project directory.
cd /lfs/turing1/0/rschaef/FieteLab-Sorscher-2022-Reproduction

# Make code inside src/ available to python.
export PYTHONPATH=.

# Determine the right W&B username.
if [[ $USER == "rschaef" ]]
then
  user="rylan"
elif [[ $USER == "mikail" ]]
then
  user="mikailkhona"
fi


# write the executed command to the slurm output file for easy reproduction
# https://stackoverflow.com/questions/5750450/how-can-i-print-each-command-before-executing
set -x

# https://docs.wandb.ai/guides/sweeps/parallelize-agents#parallelize-on-a-multi-gpu-machine
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=${gpu_id}
wandb agent ${user}/sorscher-2022-reproduction/${id}