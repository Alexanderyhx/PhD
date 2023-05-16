#!/usr/bin/env bash
#SBATCH --job-name=predfusion   # Name to appear in squeue 
#SBATCH --time       00:30:00     # Max walltime 
#SBATCH --mem        1GB
#SBATCH --account uoo03790
#SBATCH --ntasks=1

datafolder=$1
export datafolder


module purge && module load Python/3.10.5-gimkl-2022a
. venv/bin/activate





# run the model
python predoptfusion.py
