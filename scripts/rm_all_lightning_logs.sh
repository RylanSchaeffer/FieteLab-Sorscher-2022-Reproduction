#!/bin/bash

# Loop over lowercase alphabet characters
for char in {a..z}
do
    sbatch scripts/rm_lightning_log.sh "$char"
done

# Loop over uppercase alphabet characters
#for char in {A..Z}
#do
#    sbatch scripts/rm_lightning_log.sh "$char"
#done

# Loop over digits from 0 to 9
for digit in {0..9}
do
    sbatch scripts/rm_lightning_log.sh "$digit"
done
