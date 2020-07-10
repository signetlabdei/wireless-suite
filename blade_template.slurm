#!/bin/bash

#SBATCH --job-name foo
#SBATCH --output foo_output.o%j
#SBATCH --error foo_output.e%j
#SBATCH --mail-type ALL
#SBATCH --time 24:00:00
#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --mem 32G

source /nfsd/opt/anaconda3/anaconda3.sh
conda activate /home/dragomat/.conda/envs/gym-env
srun python /nfsd/signet3/dragomat/wireless-suite/wireless/scripts/foo.py