#! /bin/bash

# first job - no dependencies
jid1=$(sbatch batch_files/build_sample.sbatch | awk -F' '  '{print $NF}')
# multiple jobs can depend on a single job
jid2=$(sbatch  --dependency=afterok:$jid1 batch_files/dedup.sbatch | awk -F' '  '{print $NF}')
jid3=$(sbatch  --dependency=afterok:$jid2 batch_files/predict.sbatch | awk -F' '  '{print $NF}')
jid4=$(sbatch  --dependency=afterok:$jid3 batch_files/post_predict.sbatch | awk -F' '  '{print $NF}')

# show dependencies in squeue output:
squeue -u your_user_name
