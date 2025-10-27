#!/bin/bash
#SBATCH --job-name=DEMO_DEPENDENT
#SBATCH --partition=COMPUTE
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2000
#SBATCH --exclude=compute-0-[28]
#SBATCH --error=dependent_error.txt
#SBATCH --output=dependent_output.txt

/home/ggu7596/miniconda3/bin/python3 TRM_sparse_matrix.py $1
/home/ggu7596/miniconda3/bin/python3 compute_pi_and_current_from_TRM.py $1 $2
