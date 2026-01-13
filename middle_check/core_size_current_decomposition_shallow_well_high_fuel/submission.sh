#!/bin/bash                                                                   
#SBATCH --job-name test                                                       
#SBATCH --partition=COMPUTE 
#SBATCH --qos normal                                                            
#SBATCH --nodes=1                                                               
#SBATCH --ntasks-per-node=1                                                     
#SBATCH --mem=2000                                                              
#SBATCH --error=error.txt                                                       
#SBATCH --output=out.txt                                                        
echo started at `date`>> time.log
/home/ggu7596/miniconda3/bin/python3 run_nonequilibrium.py
echo finished at `date`>> time.log