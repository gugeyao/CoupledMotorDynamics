import sys
import numpy as np
sys.path.append('/home/ggu7596/project/optimal_control/motorsim_double_track/toy_model_v2')
from diffusion_jump_motor import DiffusionJumpMotor

parallel_jobs = 1 # no parallel jobs
steps = 400000000
shifted_distance = 6
barrier_height = 3.15
eta = 0
MC_steps = 100
well_width = 6

core_size = 3
epR = 1e3
repeated_length = 12
coupling_strength=np.float64(sys.argv[1])
k_attach_far = np.float64(sys.argv[2]) # tune how fast the reaction can happen
motor = DiffusionJumpMotor(MC_steps = MC_steps, eta = eta, k_attach_far = k_attach_far, coupling_strength = coupling_strength, shifted_distance = shifted_distance,core_size = core_size, barrier_height=barrier_height,epR = epR, well_width = well_width, repeated_length = repeated_length)
motor.parallel_run_simulation(parallel_jobs,steps)
