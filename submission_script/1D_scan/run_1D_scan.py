import sys
import numpy as np
sys.path.append('/home/ggu7596/project/optimal_control/motorsim_double_track/toy_model_v2')
from diffusion_jump_motor import DiffusionJumpMotor

parallel_jobs = 1 # no parallel jobs

steps = np.int32(sys.argv[1])
barrier_height = np.float64(sys.argv[2]) # the binding affinity of the ring to the track
k_attach_right_FTC = np.float64(sys.argv[3]) # tune how fast the reaction can happen
epR =  np.float64(sys.argv[4]) # repulsion between ring and red
E_C_track = np.float64(sys.argv[5])
gamma = 8
motor = DiffusionJumpMotor(barrier_height=barrier_height, k_attach_right_FTC = k_attach_right_FTC, epR = epR, E_C_track = E_C_track,gamma = gamma)
motor.parallel_run_simulation(parallel_jobs,steps)
