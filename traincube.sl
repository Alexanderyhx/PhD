#!/bin/bash -e
#SBATCH --licenses=matlab@uoo:1
#SBATCH --job-name   MATLAB_job   # Name to appear in squeue 
#SBATCH --time    30:00:00     # Max walltime 
#SBATCH --mem       17gb        # Max memory
#SBATCH --account uoo03698
#SBATCH --cpus-per-task=50
#SBATCH --ntasks=1
datafolder=$1
export datafolder


module load MATLAB/2022a
# Run the MATLAB script MATLAB_job.m 
matlab -nodisplay < AlexTrainandsub.m
