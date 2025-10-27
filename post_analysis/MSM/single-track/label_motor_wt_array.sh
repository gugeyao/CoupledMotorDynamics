#!/bin/bash
#SBATCH --job-name=DEMO_ARRAY
#SBATCH --partition=COMPUTE
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2000
#SBATCH --exclude=compute-0-[28]
#SBATCH --array=1-100
#SBATCH --error=error.txt
#SBATCH --output=output.txt
#SBATCH --array=1-100
echo "started at `date`" >> time.log
/home/ggu7596/miniconda3/bin/python3 label_motor_wt.py $SLURM_ARRAY_TASK_ID $1
echo "finished at `date`" >> time.log
