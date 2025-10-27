## Single-track MSM post-analysis workflow

This directory builds a continuous-time Markov State Model (MSM) from simulated single-track trajectories of the double-ring motor. The pipeline:

- Coarse-grains raw trajectory frames into discrete states using ring position and neighboring occupancy features
- Selects frequently visited “good” centers as state representatives
- Assigns every frame to its nearest center to produce labeled transitions with waiting times
- Estimates a sparse transition rate matrix (generator) from counts and residence times
- Solves for the steady-state distribution and computes the net ring current

All steps are scripted for SLURM HPC submission and can also be run manually.

### Directory layout

- `collect_unique_states.py`: scans trials to extract unique coarse-grained states and selects frequent centers
- `label_motor_wt.py`: labels each sample in a trial to the nearest center and saves `[state_label, waiting_time]`
- `TRM_sparse_matrix.py`: builds the sparse transition rate matrix and time-averaged population
- `compute_pi_and_current_from_TRM.py`: solves for steady-state distribution and computes current
- `a_full_workflow.sh`: submits the full 3-stage workflow on SLURM (collect → label array → TRM+pi)
- `demo_1arg.sh`, `label_motor_wt_array.sh`, `TRM_compute_pi.sh`: SLURM wrappers

### Prerequisites

- Python 3.8+ with packages: `numpy`, `scipy`, `deeptime`
  - Example: `pip install numpy scipy deeptime`
- SLURM cluster (optional, for the array job workflow)
- Simulation output tree at `PATH` with per-trial text files under `PATH/transition_data/`
  - Expected files: `0001.txt`, `0002.txt`, ..., up to the configured number of trials

### Input data expectations

Each trial file `PATH/transition_data/%04d.txt` is a whitespace-delimited text file containing, at minimum, the following columns used by the scripts:

- Column 2: waiting time for the current sample (used in rate estimation)
- Column 4: ring 1 position index (integer)
- Columns 5–8: occupancy indicators for the four motif “C” groups near the ring (integers)

The coarse-grained state for a frame is defined as `[ring_pos1, C_left, C_center, C_right]` where the three `C_*` values are the occupancy values of the three motif groups nearest to the ring position, with periodic wrap-around over motifs.

Default geometry assumptions in the scripts:

- Number of motifs: `num_motif = 4`
- Beads per motif (track period): `length_motif = 12`
- Total beads per ring track: `num_motif * length_motif = 48`

If your simulation uses different values, update the constants in `collect_unique_states.py` and `label_motor_wt.py` accordingly, and pass the correct `length_motif` as an argument to the last stage (see below).

### End-to-end (SLURM) workflow

1) Edit `a_full_workflow.sh`:

- `arg1`: absolute path to your dataset (the folder containing `transition_data/`)
- `arg2`: `length_motif` (e.g., `12`)

2) Submit the chained jobs:

```bash
bash a_full_workflow.sh
```

The script will:

- Submit `collect_unique_states.py` once to gather unique states and select frequent centers
- Submit an array job `label_motor_wt_array.sh` over trials `1-100` to label frames and save `[state_label, waiting_time]`
- After the array completes, run `TRM_sparse_matrix.py` and `compute_pi_and_current_from_TRM.py`

Job IDs are appended to `PATH/job_ids_post_process.log`. Adjust SLURM directives (partition, memory, python path) in the wrapper scripts if needed for your cluster.

### Manual workflow (no SLURM)

Assuming `PATH=/absolute/path/to/your/dataset` and trials `1..100`:

1) Extract unique states and select “good” centers (threshold count > 10 by default):

```bash
python3 collect_unique_states.py $PATH
```

Outputs:

- `$PATH/unique_states_CG.txt` (columns: ring_pos, C_left, C_center, C_right, count)
- `$PATH/good_centers_CG.txt` (same columns as above, filtered to count > 10)

2) Label each trial and save `[state_label, waiting_time]`:

```bash
# Single trial
python3 label_motor_wt.py 1 $PATH

# All trials (bash loop)
for i in $(seq -w 1 100); do
  python3 label_motor_wt.py $((10#$i)) $PATH
done
```

Outputs per trial:

- `$PATH/transition_data/%04d_labeled.txt` with two columns: `state_label` (int), `waiting_time` (float)

3) Build sparse transition rate matrix and empirical population:

```bash
python3 TRM_sparse_matrix.py $PATH
```

Outputs:

- `$PATH/TRM_CG.txt` with non-zero triplets `(row, col, rate)`
- `$PATH/Population_CG.txt` empirical time fractions per state

4) Compute steady-state distribution and net current:

```bash
# pass length_motif (e.g., 12)
python3 compute_pi_and_current_from_TRM.py $PATH 12
```

Outputs:

- `$PATH/computed_population_CG.txt` steady-state probabilities (linear solver result)
- `$PATH/current_MSM_CG.txt` scalar current

### How the MSM is built (details)

1) State definition and center selection (`collect_unique_states.py`):

- For each trial file, map frames to coarse-grained vectors `[ring_pos1, C_left, C_center, C_right]` using ring position and the three nearest motif occupancy values with periodic boundary conditions
- Aggregate over trials; compute unique state vectors and their counts
- Keep “good centers” with `count > 10` as cluster representatives (edit the threshold if needed)

2) Frame-to-center assignment (`label_motor_wt.py`):

- Loads `good_centers_CG.txt` and uses `deeptime.clustering.ClusterModel` with fixed `cluster_centers` to assign each frame to its nearest center
- Saves `[state_label, waiting_time]` for each consecutive frame; labels are 0-based indices into the center list

3) Generator estimation (`TRM_sparse_matrix.py`):

- For each trial, counts transitions from state `i` to `j` across consecutive labeled frames
- Computes the total waiting time accumulated in each origin state `i`
- Estimates off-diagonal rates as `q_ij = (counts_ij) / (total_wait_i)`; then sets diagonal to `q_ii = -∑_{j≠i} q_ij` so each row sums to zero
- Writes non-zero entries to `TRM_CG.txt` as triplets and time-averaged occupancy to `Population_CG.txt`

4) Steady-state and current (`compute_pi_and_current_from_TRM.py`):

- Builds a dense rate matrix from the triplets and solves `Q^T π = 0` using a small forcing vector and normalization
- Computes net current by summing `Δring_pos × q_ij × π_i` over all non-zero edges, with periodic wraparound over `num_sets = 4` track copies and bead period `length_motif`

### Configuration knobs

- Number of trials: set in scripts (default `100`); update loops if you have a different count
- Good-center threshold: `count > 10` in `collect_unique_states.py`; lower it for sparser data
- Geometry: `num_motif` and `length_motif` in the python scripts; also pass `length_motif` to the final step
- Grouping for TRM: `groups` in `TRM_sparse_matrix.py` (default `1`); can be used to average blockwise
- Python path and SLURM settings: edit the wrappers to match your environment

### Outputs summary

- `$PATH/unique_states_CG.txt`: candidate coarse-grained states with visit counts
- `$PATH/good_centers_CG.txt`: final state centers used for labeling
- `$PATH/transition_data/%04d_labeled.txt`: per-trial labeled sequence with waiting times
- `$PATH/TRM_CG.txt`: sparse generator triplets `(row, col, rate)`
- `$PATH/Population_CG.txt`: empirical time fractions
- `$PATH/computed_population_CG.txt`: steady-state `π` from the linear solve
- `$PATH/current_MSM_CG.txt`: net current scalar

### Troubleshooting

- Zero-waiting-time states reported in `TRM_sparse_matrix.py`: indicates some labeled states were never occupied as origins; confirm labeling and trial completeness
- Linear solver issues: ensure `TRM_CG.txt` covers a connected component; if `det(Q^T)` prints pathological values, verify data coverage and thresholds
- Mismatch of bead period: ensure you pass the correct `length_motif` to `compute_pi_and_current_from_TRM.py` and that geometry constants match your simulation
- Missing `deeptime`: `pip install deeptime` or adjust to a nearest-neighbor assignment if preferred


