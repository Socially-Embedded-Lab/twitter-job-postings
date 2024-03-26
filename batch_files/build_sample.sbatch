#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main			### specify partition name where to run a job.
#SBATCH --time 7-00:00:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name build_sample			### name of the job
#SBATCH --output build_job_offer_data.out			### output log for running job - %J for job number

#SBATCH --gpus=1				### number of GPUs, allocating more than 1 requires IT team's permission
#SBATCH --mem=24G				### ammount of RAM memory, allocating more than 60G requires IT team's permission

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start your code below ####
module load anaconda				### load anaconda module (must be present when working with conda environments)
source activate my_env			### activate a conda environment, replace my_env with your conda environment
cd ~/twitter-job-postings-private/data/scripts
python build_job_offer.py
