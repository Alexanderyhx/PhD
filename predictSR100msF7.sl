#!/usr/bin/env bash
#SBATCH --job-name=spikeratepredoptwwknn   # Name to appear in squeue 
#SBATCH --time       10:00:00     # Max walltime 
#SBATCH --mem        5gb        # Max memory
#SBATCH --account uoo03790
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

datafolder=$1
export datafolder


module purge && module load Python/3.10.5-gimkl-2022a
. venv/bin/activate

# run the model
python spikeratepredoptwwknn.py
