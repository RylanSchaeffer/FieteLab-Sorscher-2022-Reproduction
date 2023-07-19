#!/bin/bash

cd /om2/user/rylansch/FieteLab-Sorscher-2022-Reproduction/lightning_logs

# Loop over the files in the directory
for file in *
do
    echo "$file"
    sbatch ../scripts/rm_lightning_log.sh "$file"
done
