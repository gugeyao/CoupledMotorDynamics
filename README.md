## Jump-Diffusion Model for Coupled Molecular Motor

Simulation for a diffusion–jump molecular motor on two coupled periodic tracks. The main implementation is in `diffusion_jump_motor.py` with a high-level API and Slurm-friendly submission scripts.

### Modules
- `diffusion_jump_motor.py`
  - `diffusion_jump_motor`: Low-level class with full state and integrator.
  - `DiffusionJumpMotor`: High-level wrapper for convenient parameterization and runs.
- `submission_script/`
  - `run_spring_coupling.py`: CLI entry to run a single simulation (2 args).
  - `submission_spring_coupling.sh`: Submits a grid of runs via Slurm.
  - `demo_2argv.sh`: Slurm job template used by the submission script.

### Requirements
- Python 3.8+
- Packages: `numpy`, `numba`, `joblib`

Install:
```bash
python -m pip install --upgrade pip
python -m pip install numpy numba joblib
```


```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### Quick start (Python API)
```python
from diffusion_jump_motor import DiffusionJumpMotor

motor = DiffusionJumpMotor(
    MC_steps=100,
    barrier_height=3.15,
    well_width=6,
    repeated_length=12,
    num_motifs=4,
    epR=1e3,
    shifted_distance=6,
    coupling_strength=0.01,
    coupling_center=0.0,
    k_attach_far=2e-4,
    eta=0,
    core_size=3,
    gamma=6,
)

motor.parallel_run_simulation(parallel_jobs=1, steps=1_000_000)
```

Outputs are written to `simulation_data/` and `transition_data/` in the working directory. If the run is long enough, `hopping_rates.txt` is generated.

### Run via the provided script
`run_spring_coupling.py` takes two arguments: `coupling_strength` and `k_attach_far`.
```bash
python submission_script/run_spring_coupling.py 0.01 0.0002
```

You can edit defaults like `steps`, `parallel_jobs`, `shifted_distance`, etc., at the top of `submission_script/run_spring_coupling.py`.

### Run many simulations simultaneously (Slurm)
1) Change to the submission directory:
```bash
cd submission_script
```

2) Launch the sweep:
```bash
bash submission_spring_coupling.sh
```

What happens:
- The script sweeps over arrays `coupling_strengths` and `k_attach_fars`.
- For each combination, a work directory is created and `sbatch demo_2argv.sh` is called.
- Slurm job IDs are appended to `job_ids.log` in this directory.

Cluster notes:
- Adjust `#SBATCH` settings and the Python interpreter line in `demo_2argv.sh` to match your environment (e.g., use `python3 program argv1 argv2` or your conda/venv python path).

### Key files written
- `simulation_data/*.txt`: trajectory snapshots and cycle counters (per replica)
- `transition_data/*.txt`: coarse-grained transition records (per replica)
- `hopping_rates.txt`: aggregate hopping rates (when simulation is long enough)

### Tips
- To change the parameter grid of the sweep, edit the arrays at the top of `submission_spring_coupling.sh`.
- For single long runs, prefer running `run_spring_coupling.py` directly with your desired arguments.



### Copyright
Copyright (c) 2025 Geyao Gu. All rights reserved.
