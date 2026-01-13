# Equilibrium check for the toy model
import sys
sys.path.append('/home/ggu7596/project/optimal_control/motorsim_double_track/toy_model_v2')
from diffusion_jump_motor import DiffusionJumpMotor

# motor object
# keep the same parameters as the old simulation
steps = 400000000
MC_steps = 100
eta = 0.1
k_attach_far = 0.02
coupling_strength = 0
shifted_distance = 6
core_size = 1
barrier_height = 1.15
motor = DiffusionJumpMotor(MC_steps = MC_steps, eta = eta, k_attach_far = k_attach_far, coupling_strength = coupling_strength, shifted_distance = shifted_distance,core_size = core_size, barrier_height=barrier_height)
motor.parallel_run_simulation(parallel_jobs=1, steps=steps)
