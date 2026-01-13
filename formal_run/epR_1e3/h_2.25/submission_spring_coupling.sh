coupling_strengths=(0 0.001 0.0018 0.0032 0.0056 0.01 0.018 0.032 0.056 0.1 0.18 0.32 0.56 1)
k_attach_fars=(0.00002 0.0002 0.002 0.02)
# Outer loop over k_attach_right_FTC values
workpath=$(pwd)
program_name="run_spring_coupling.py"
for k_attach_far in "${k_attach_fars[@]}"; do
    subworkpath="$workpath/k_attach_far_$k_attach_far"
    mkdir -p "$subworkpath"

    # Loop over spring strengths
    for coupling_strength in "${coupling_strengths[@]}"; do
        # Define trial name based on current parameters
        # Create work directory
        work="$subworkpath/coupling_strength_$coupling_strength"
        mkdir -p "$work"
        
        # Copy the program and create the job script
        cp "$workpath/$program_name" "$work/$program_name"
        sed -e "s/DEMO/$trial_name/g" \
            -e "s/program/$program_name/g" \
            -e "s/argv1/$coupling_strength/g" \
            -e "s/argv2/$k_attach_far/g" \
            "$workpath/demo_2argv.sh" > "$work/demo.sh"

        # Change to the work directory and submit the job
        cd "$work" || { echo "Failed to change directory to $work"; exit 1; }
        job_id=$(sbatch demo.sh)
        echo "$job_id"
        echo "$job_id" >> "$workpath/job_ids.log"
    done
done
