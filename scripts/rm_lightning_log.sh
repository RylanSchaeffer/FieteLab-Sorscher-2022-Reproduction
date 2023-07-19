#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=01:00:00

# Retrieve the input string from the command line argument
string_prefix="$1"

echo $string_prefix

# Change to the desired directory containing the files
cd /om2/user/rylansch/FieteLab-Sorscher-2022-Reproduction/lightning_logs

# Remove files starting with the input string
rm -rf "${string_prefix}"

