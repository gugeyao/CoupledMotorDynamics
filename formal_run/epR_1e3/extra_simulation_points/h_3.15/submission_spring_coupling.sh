coupling_strengths=(0 0.1)
k_attach_fars=(0.00002 0.0002)
num_repeats=20
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
        for i in $(seq 1 $num_repeats); do
            work2="$work/repeat_$i"
            mkdir -p "$work2"
            # Copy the program and create the job script
            cp "$workpath/$program_name" "$work2/$program_name"
            sed -e "s/DEMO/$trial_name/g" \
                -e "s/program/$program_name/g" \
                -e "s/argv1/$coupling_strength/g" \
                -e "s/argv2/$k_attach_far/g" \
                "$workpath/demo_2argv.sh" > "$work2/demo.sh"

            # Change to the work directory and submit the job
            cd "$work2" || { echo "Failed to change directory to $work2"; exit 1; }
            job_id=$(sbatch demo.sh)
            echo "$job_id"
            echo "$job_id" >> "$workpath/job_ids.log"
        done
    done
done
