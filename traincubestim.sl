#!/bin/bash -e
#SBATCH --licenses=matlab@uoo:1
#SBATCH --job-name   MATLAB_job   # Name to appear in squeue 
#SBATCH --time      00:30:00     # Max walltime 
#SBATCH --mem        5gb        # Max memory
#SBATCH --account uoo03698 





#         SBATCH --array=1-x

# sbatch --array=1-$(wc -l combo.csv) slurm/predictopts.sl
# sbatch --array=4 slurm/predictopt.sl
# run as 
# sbatch slurm/predictfusion.sl
#datafolder=pwd

datafolder=$1
export datafolder



module load MATLAB/2022a
# Run the MATLAB script MATLAB_job.m 
matlab -nodisplay < stim5AlexTrainandsub.m 
