#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # one node
#SBATCH --mem=20G               # RAM
#SBATCH --gres=gpu:1
#SBATCH --constraint=16GB
#SBATCH --time=7-00:00:00         # total run time limit (D-HH:MM:SS)
#SBATCH --mail-type=FAIL

id=${1}

source ~/.bashrc

module load openmind/cuda/11.2
module load openmind/cudnn/11.5-v8.3.3.40

# Activate conda environment.
conda activate sorscher_reproduction

cd /om2/user/rylansch/FieteLab-Sorscher-2022-Reproduction

# Activate pip virtual environment.
source sorscher_reproduction_venv/bin/activate

export PYTHONPATH=.


if [[ $USER == "rylansch" ]]
then
  user="rylan"
elif [[ $USER == "mikail" ]]
then
  user="mikailkhona"
fi



# write the executed command to the slurm output file for easy reproduction
# https://stackoverflow.com/questions/5750450/how-can-i-print-each-command-before-executing
set -x

wandb agent ${user}/sorscher-2022-reproduction/${id}