#!/usr/bin/env bash
#SBATCH --job-name=spikerate  # Name to appear in squeue    
#SBATCH --time       10:00:00     # Max walltime 
#SBATCH --mem        5gb 
#SBATCH --account uoo03790
#SBATCH --ntasks=1

datafolder=$1
export datafolder


module purge && module load Python/3.10.5-gimkl-2022a
. venv/bin/activate



# run the model
python spikerate2sF7.py

