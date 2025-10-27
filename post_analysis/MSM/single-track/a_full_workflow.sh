#!/bin/bash

# Define the arguments
arg1="/home/ggu7596/project/optimal_control/motorsim_double_track/data/week07year2025/Diffusion_Jump_spring_coupling_barrier_height_0.702_revised_para_core/k_attach_right_FTC_0.002/spring_strength_0.1"
arg2="12" #number of beads
# Submit job 1
job1_id=$(sbatch `echo demo_1arg.sh collect_unique_states.py $arg1` | awk '{print $4}')
echo "Workflow submitted. Job 1 ID: $job1_id"
echo "$job1_id" >> $arg1/job_ids_post_process.log
# Submit jobs 2.1 through 2.100 as an array
job2_array_id=$(sbatch `echo -d afterany:$job1_id label_motor_wt_array.sh $arg1`| awk '{print $4}')
echo "Workflow submitted. Job 2 ID: $job2_array_id"
echo "$job2_array_id" >> $arg1/job_ids_post_process.log
# Submit job 3 with dependency on the job2 array job
job3_id=$(sbatch `echo -d afterany:$job2_array_id TRM_compute_pi.sh $arg1 $arg2` | awk '{print $4}')
echo "Workflow submitted. Job 3 ID: $job3_id"
echo "$job3_id" >> $arg1/job_ids_post_process.log
