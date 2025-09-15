#!/usr/bin/env python3
# this version has more tunable parameters on the potential.
# -*- coding: utf-8 -*-

"""
1/16/2024

@author: gugeyao
Run brute-force dynamics on Diffusion-Jump model.
"""
import os
import sys
import numba as nb
import time
import numpy as np
import warnings
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')

nopython = True

# =============================================================================
# POTENTIAL FUNCTIONS
# =============================================================================

@nb.jit(nopython=nopython)
def potential_bare_track_(x=np.array([]),length =0,barrier_height = 0):
    """Bare track potential using sinusoidal function."""
    indices = np.where((x >= -length/2) & (x<= length/2))[0]
    U = np.zeros_like(x)+barrier_height/2
    U[indices] = barrier_height/2*np.sin(2*np.pi/length*(x[indices]-length/4))
    return U

@nb.jit(nopython=nopython)
def potential_LJ_(epA = 0, epR = 1e4, dr_CAT=2, sigma = 1):
    """
    Calculate Lennard-Jones potential for repulsive interactions.
    
    Args:
        epA (float): Attractive strength parameter (default: 0)
        epR (float): Repulsive strength parameter (default: 1e4)
        dr_CAT (float): Distance to catalytic site (default: 2)
        sigma (float): LJ length scale parameter (default: 1)
        
    Returns:
        float: Lennard-Jones potential energy value
        
    Formula:
        U = 4 * (-epA * (sigma/dr_CAT)^6 + epR * (sigma/dr_CAT)^12)
        
    Notes:
        - Uses standard 6-12 Lennard-Jones form
        - epA controls attractive interactions (typically 0 for barriers)
        - epR controls repulsive interactions (creates barriers)
        - dr_CAT should be positive distance
    """
    U = 4*(-epA*(sigma/dr_CAT)**6+epR*(sigma/dr_CAT)**12)
    return U

@nb.jit(nopython=nopython)
def potential_spring_(dr=np.array([]),k= 0,x0 = 0,cross_distance = 10):
    """Spring potential for coupling between particles."""
    delta_x = np.sqrt((dr)**2+cross_distance**2)
    U = 0.5 * k *(delta_x - x0)**2
    return U

# =============================================================================
# FORCE FUNCTIONS (ANALYTICAL - OPTIMIZED)
# =============================================================================

@nb.jit(nopython=nopython)
def force_bare_track_(x_elementary=np.array([]),length =0,barrier_height = 0):
    """Force from bare track potential using analytical derivative."""
    # PBC: x_elementary is already PBC-handled (elementary_shifted_x)
    indices = np.where((x_elementary >= -length/2) & (x_elementary <= length/2))[0]
    f = np.zeros_like(x_elementary)
    f[indices] = -barrier_height*np.pi/length * np.cos(2*np.pi/length*(x_elementary[indices]-length/4))
    return f

@nb.jit(nopython=nopython)
def force_LJ_(epA = 0, epR = 1e4, dr_CAT=2, sigma = 1):
    """
    Calculate force from Lennard-Jones potential using analytical derivative.
    
    Args:
        epA (float): Attractive strength parameter (default: 0)
        epR (float): Repulsive strength parameter (default: 1e4)
        dr_CAT (float): Distance to catalytic site (default: 2)
        sigma (float): LJ length scale parameter (default: 1)
        
    Returns:
        float: LJ force value (negative gradient of potential)
        
    Formula:
        F = -dU/dr = -4 * (6 * epA * sigma^6 / dr_CAT^7 - 12 * epR * sigma^12 / dr_CAT^13)
        
    Notes:
        - Force is the negative gradient of the LJ potential
        - dr_CAT should be PBC-handled before calling (from compute_dCAT)
        - Positive force = repulsive, negative force = attractive
        - Uses analytical derivative for computational efficiency
    """
    f = -4 * (6 * epA * sigma**6 / dr_CAT**7 - 12 * epR * sigma**12 / dr_CAT**13)
    return f

@nb.jit(nopython=nopython)
def force_spring_(dr=np.array([]),k = 0,x0 = 0,cross_distance = 10):
    """Force from spring potential using analytical derivative."""
    # PBC: dr is already PBC-handled (from compute_dr)
    # For spring potential U = 0.5 * k * (sqrt(dr^2 + cross_distance^2) - x0)^2
    # Force on particle 1: F1 = -dU/dx1 = -dU/dr * dr/dx1 = -dU/dr * (-1) = dU/dr
    # Since dr = x2 - x1, we have dr/dx1 = -1
    # So F1 = dU/dr = k * (sqrt(dr^2 + cross_distance^2) - x0) * dr / sqrt(dr^2 + cross_distance^2)
    delta_x = np.sqrt((dr)**2 + cross_distance**2)
    # Avoid division by zero by adding small epsilon
    epsilon = 1e-10
    f = k * (delta_x - x0) * dr / (delta_x + epsilon)
    return f

@nb.jit(nopython=True)
def force_LJ_optimized_sparse_precomputed(const1, const2, dx_CAT, choices):
    """
    Optimized LJ force computation using pre-computed constants.
    
    Only computes forces for occupied sites to improve performance.
    
    Args:
        const1 (float): Pre-computed constant: -24 * epA * sigma^6
        const2 (float): Pre-computed constant: 48 * epR * sigma^12
        dx_CAT (np.ndarray): Distances to catalytic sites, shape (n, num_motifs)
        choices (np.ndarray): Occupancy array, shape (n, num_motifs), 1=occupied, 0=unoccupied
        
    Returns:
        np.ndarray: Total LJ force for each particle, shape (n,)
        
    Performance:
        - Only computes forces for occupied sites (choices == 1)
        - Uses pre-computed constants to avoid redundant calculations
        - Significantly faster than computing all sites then masking
        
    Notes:
        - Requires pre-computed constants from initialization
        - Only processes occupied sites (choices[i,j] == 1)
        - Accumulates forces for each particle across all occupied sites
        - Uses the same mathematical formula as force_LJ_ but optimized
    """
    n, num_motifs = dx_CAT.shape
    f_LJ_sum = np.zeros(n)
    
    # Loop through all particles and sites
    for i in range(n):
        for j in range(num_motifs):
            if choices[i, j] == 1:  # Only compute for occupied sites
                dr_CAT = dx_CAT[i, j]
                dr7 = dr_CAT**7
                dr13 = dr_CAT**13
                f = const1 / dr7 + const2 / dr13
                f_LJ_sum[i] += f
    
    return f_LJ_sum

# =============================================================================
# ANALYSIS AND UTILITY FUNCTIONS
# =============================================================================

def hopping_rates(num_trial):
    """
    Calculate hopping rates from simulation data files.
    
    Args:
        num_trial (int): Number of trial simulations to analyze
        
    Returns:
        tuple: (right_cyc_rate_mean, right_cyc_rate_err, left_cyc_rate_mean, left_cyc_rate_err)
            - right_cyc_rate_mean (float): Average rightward hopping rate
            - right_cyc_rate_err (float): Standard error of rightward rate
            - left_cyc_rate_mean (float): Average leftward hopping rate  
            - left_cyc_rate_err (float): Standard error of leftward rate
            
    Notes:
        - Reads data from "simulation_data" folder
        - Expects files named "0001.txt", "0002.txt", etc.
        - Uses first and last rows of each file for rate calculation
        - Calculates rates as (final_cycles - initial_cycles) / total_time
    """
    a_folder = "simulation_data"
    right_cyc_rates = np.zeros(num_trial)
    left_cyc_rates = np.zeros(num_trial)
    for k in range(num_trial):
        file = "%04d.txt"%(k+1)
        data = np.loadtxt(open(a_folder+"/"+file))[[0,-1],:]
        time = data[:,0]
        r1 = data[:,3]  # right cycles for particle 1
        l1 = data[:,4]  # left cycles for particle 1
        r2 = data[:,5]  # right cycles for particle 2
        l2 = data[:,6]  # left cycles for particle 2
        delta_r1 = r1[-1] - r1[0]
        delta_l1 = l1[-1] - l1[0]
        delta_r2 = r2[-1] - r2[0]
        delta_l2 = l2[-1] - l2[0]
        delta_t = time[-1] - time[0]
        right_cyc_rate = (delta_r1 +delta_r2)/2/delta_t
        left_cyc_rate = (delta_l1 +delta_l2)/2/delta_t
        right_cyc_rates[k] = right_cyc_rate
        left_cyc_rates[k] = left_cyc_rate
    right_cyc_rate_mean = np.mean(right_cyc_rates)
    right_cyc_rate_err = np.std(right_cyc_rates)/np.sqrt(num_trial)
    left_cyc_rate_mean = np.mean(left_cyc_rates)
    left_cyc_rate_err = np.std(left_cyc_rates)/np.sqrt(num_trial)
    return right_cyc_rate_mean,right_cyc_rate_err,left_cyc_rate_mean,left_cyc_rate_err

# =============================================================================
# MAIN DIFFUSION JUMP MOTOR CLASS
# =============================================================================

class diffusion_jump_motor(object):    
    """
    Diffusion-Jump Motor Simulation Class
    
    Usage:
    1. initialize the system
    2. run the simulation
    3. analyze the results
    
    Author: Gugeyao
    """
    
    # =============================================================================
    # INITIALIZATION AND SETUP
    # =============================================================================
    
    def __init__(self):
        """Initialize the diffusion jump motor with default parameters."""
        # bare-track potential parameters
        self.well_width = 6 # repeated length of the sin function
        self.barrier_height = 3.15 # the barrier height of the sinx well
        self.repeated_length = 12
        self.num_motifs = 4 # number of periodic wells

        # blocked potential parameters
        self.epA = 0
        self.epR = 1e4
        self.sigma = 1 # radii of the repulsion

        # system parameters
        self.beta = 2
        self.m = 12
        self.n = 100 #number of replicas (traj.)
        self.dt = 0.005 # to prevent too large step
        self.gamma = 6
        self.cycle = 10000 # the length of generating random numbers

        # coupling parameters
        self.k = 0 # spring strength
        self.x0 = 0 # spring center
        self.cross_distance = 10 # the distance between the double potential
        self.shifted_distance = 0 # the shifted phase between the two rings

        # rate parameters
        self.k_attach_far = 2e-4
        self.center_attach = 4.6
        self.spread_attach = 0.001 # as small as possible to avoid 
        self.k_detach_far = 1.5e-4
        self.eta = 1 # how much the detachment rate changes as a function of distance to the ring follows the LJ potential.
        self.MC_steps = 100
        
        # core detection parameter (tunable core diameter around each binding position)
        self.core_size = 1.0

    def initialize_system(self):        
        """Initialize the simulation system with all required variables."""
        # VRORV integrator parameters (matches C++ implementation)
        # initialization parameters
        self.tot_length = self.num_motifs * self.repeated_length
        self.x1 = np.random.uniform(0,self.tot_length,(self.n)) # initialize the positions of the particles
        self.x2 = np.array(self.x1)+np.random.uniform(-self.tot_length/((1+self.k*self.beta*10)*2),self.tot_length/((1+self.k*self.beta*10)*2),(self.n)) # not causing a big energy difference
        self.boundary()
        
        # Pre-allocate arrays for compute_shifted_x_and_elementary_x optimization (must be before the call)
        self.shifted_x1 = np.zeros(self.n)
        self.shifted_x2 = np.zeros(self.n)
        self.elementary_shifted_x1 = np.zeros(self.n)
        self.elementary_shifted_x2 = np.zeros(self.n)
        
        self.compute_shifted_x_and_elementary_x()
        self.compute_dr()
        # Boltzmann distribution: p ~ N(0, sqrt(m/beta)) = N(0, sqrt(m*kB*T))
        self.p1 = np.random.normal(0,np.sqrt(self.m/self.beta),(self.n))
        self.p2 = np.random.normal(0,np.sqrt(self.m/self.beta),(self.n))
        self.f1 = np.zeros((self.n))
        self.f2 = np.zeros((self.n))
        
        self.x1_state_old = np.zeros((self.n))
        self.x2_state_old = np.zeros((self.n))
        self.x1_state_new = np.zeros((self.n))
        self.x2_state_new = np.zeros((self.n))

        # analysis parameters
        self.right_cycles_x1 = np.zeros((self.n))
        self.right_cycles_x2 = np.zeros((self.n))
        self.left_cycles_x1 = np.zeros((self.n))
        self.left_cycles_x2 = np.zeros((self.n))

        # well indices of the current particles
        self.well_idx_1 = np.zeros((self.n))
        self.well_idx_2 = np.zeros((self.n))
        
        #  initial as unblocked state
        self.choices_1 = np.zeros((self.n,self.num_motifs)) # each choice can be blocked or unblocked, 0: unblocked, 1:  blocked
        self.choices_2 = np.zeros((self.n,self.num_motifs)) # each choice can be blocked or unblocked, 0: unblocked, 1:  blocked
        
        # Precompute cycle detection constants for efficiency
        self.right_jump = self.repeated_length
        self.left_jump = -self.repeated_length
        self.right_wrap = (1 - self.num_motifs) * self.repeated_length
        self.left_wrap = (-1 + self.num_motifs) * self.repeated_length
        
        # Pre-allocate tiled arrays for compute_dCAT optimization
        self.tiled_x1 = np.zeros((self.n, self.num_motifs))
        self.tiled_x2 = np.zeros((self.n, self.num_motifs))
        
        # Pre-compute LJ force constants for efficiency
        self.LJ_const1 = -4 * 6 * self.epA * self.sigma**6
        self.LJ_const2 = -4 * (-12) * self.epR * self.sigma**12

        # Precompute constants for efficiency
        self.dt_m = self.dt / self.m # damping coefficient
        self.exp_gamma_dt = np.exp(-self.gamma * self.dt_m)  # Matches C++: exp(-gamma*dt/m)
        self.sigma_thermal = np.sqrt(self.m * (1 - self.exp_gamma_dt**2) / self.beta)  # Matches C++: sqrt(m*(1-exp(-2*gamma*dt/m))/beta)
        
        # =============================================================================
        # MEMORY PREALLOCATION FOR PERFORMANCE OPTIMIZATION
        # =============================================================================
        
        # Pre-allocate random number arrays for main simulation loop
        self.rand1 = np.zeros((self.cycle, self.n))
        self.rand2 = np.zeros((self.cycle, self.n))
        
        # Pre-allocate Monte Carlo temporary arrays
        self.k_attach_eligible = np.zeros((self.n, self.num_motifs))
        self.k_detach_eligible = np.zeros((self.n, self.num_motifs))
        self.unblocked_mask = np.zeros((self.n, self.num_motifs), dtype=bool)
        self.blocked_mask = np.zeros((self.n, self.num_motifs), dtype=bool)
        self.flip_mask = np.zeros((self.n, self.num_motifs), dtype=bool)
        self.n_attach = np.zeros((self.n, self.num_motifs), dtype=np.int32)
        self.n_detach = np.zeros((self.n, self.num_motifs), dtype=np.int32)
        
        # Pre-allocate state tracking arrays
        self.x1_in_core = np.zeros(self.n, dtype=bool)
        self.x2_in_core = np.zeros(self.n, dtype=bool)
        self.state_changes = np.zeros(self.n, dtype=bool)
        
        # Pre-allocate trajectory array for output
        traj_size = 1 + 4*self.n + 4*self.n + 2*self.n*self.num_motifs  # time + x1,x2 + p1,p2 + cycles + choices
        self.traj_array = np.zeros(traj_size)
        
        # Pre-allocate core state arrays
        self.core_old = np.zeros((self.n, 1 + self.num_motifs))  # x1_state + choices_1
        self.core_new = np.zeros((self.n, 1 + self.num_motifs))  # x1_state + choices_1
        
        # Pre-allocate waiting time array
        self.wt_array = np.full(self.n, self.dt)
        
        # Pre-allocate integer position arrays
        self.x1_int = np.zeros(self.n, dtype=np.int32)
        self.x2_int = np.zeros(self.n, dtype=np.int32)
        
        self.CAT_position()
        self.BIND_position()
        self.force_calculation()
        self.print_out_system_info()
        
    def print_out_system_info(self):
        """Print system information and parameters."""
        ##############print info###############
        print("========System info=========")
        print("beta = "+str(self.beta))
        print("m = "+str(self.m))
        print("dt = "+str(self.dt))
        print("gamma = "+str(self.gamma))
        print("integrator = VRORV (Langevin dynamics - matches C++ implementation)")

        print("========Potential info=========")
        print("barrier height = "+str(self.barrier_height))
        print("well width = "+str(self.well_width))
        print("repeated length = "+str(self.repeated_length))
        print("num of motifs = "+str(self.num_motifs))
        print("epA = "+str(self.epA))
        print("epR = "+str(self.epR))
        
        print("========Simulation info=========")
        print("number of replicas = "+str(self.n))
        print("MC frequency = "+str(self.MC_steps))
        print("")
        
        print("========Coupling info=========")
        print("spring strength = "+str(self.k))
        print("spring center = "+str(self.x0))
        print("shifted distance = "+str(self.shifted_distance))
        print("cross distance = "+str(self.cross_distance))
        print("")
        
        print("========phenmenlogical rates==========")
        print("k_attach_far = "+str(self.k_attach_far))
        print("center_attach = "+str(self.center_attach))
        print("spread_attach = "+str(self.spread_attach))
        print("k_detach_far = "+str(self.k_detach_far))
        print("eta = "+str(self.eta))
        print("core size (diameter) = "+str(self.core_size))
        
        sys.stdout.flush()
    
    def explain_VRORV_integrator(self):
        """Explain the VRORV integrator and its advantages."""
        print("\n=== VRORV Integrator Explanation ===")
        print("VRORV is a splitting method for Langevin dynamics:")
        print("V = Velocity update (quarter time step)")
        print("R = Position update (half time step)")
        print("O = Ornstein-Uhlenbeck (thermalization)")
        print("R = Position update (half time step)")
        print("V = Velocity update (quarter time step)")
        print("\nAdvantages:")
        print("- Matches C++ implementation exactly")
        print("- Good numerical stability")
        print("- Proper thermalization")
        print("- Symplectic structure")
        print("- Time-reversible")
        print("=" * 40)

    # =============================================================================
    # UTILITY AND HELPER FUNCTIONS
    # =============================================================================
    
    def check_nan(self,x,string):
        """Check for NaN values and exit if found."""
        if np.any(np.isnan(x)):
            print(string+" contains NaN values")
            sys.exit()

    def Fermi_function(self,x,center,spread,k_right=1, k_left=0):
        """Fermi function for rate calculations."""
        return (k_right-k_left)*(1/(1+np.exp((-x+center)/spread)))+k_left 

    # =============================================================================
    # RATE CALCULATION FUNCTIONS
    # =============================================================================
    
    def k_attach_r(self,x):
        """Calculate attachment rate as a function of position."""
        k = self.Fermi_function(x,self.center_attach,self.spread_attach,self.k_attach_far, 0)
        #k[x < 2 + self.well_width/2] = 0 # prevent the particle's potential drastically change when it is on the binding site.
        return k

    def k_detach_r(self,x):
        """Calculate detachment rate as a function of position."""
        U_LJ = potential_LJ_(self.epA, self.epR, x, self.sigma)
        k = self.k_detach_far* np.exp(self.beta*U_LJ*self.eta)
        return k
    
    # =============================================================================
    # POSITION AND BOUNDARY CALCULATIONS
    # =============================================================================
    
    def BIND_position(self):
        """Calculate binding site positions for both tracks."""
        self.BIND1 = np.arange(0,self.tot_length,self.repeated_length,dtype =np.int32)
        self.BIND2 = np.arange(self.shifted_distance,self.shifted_distance+self.tot_length,self.repeated_length,dtype =np.int32)
        self.boundary_single(self.BIND1)
        self.boundary_single(self.BIND2)
        # Tiled positions for efficient distance computation (shape: n x num_motifs)
        self.BIND1_pos = np.tile(self.BIND1,(self.n,1))
        self.BIND2_pos = np.tile(self.BIND2,(self.n,1))
        
    def CAT_position(self):
        """Calculate catalytic site positions for both tracks."""
        CAT1 =np.arange(2,2+self.tot_length,self.repeated_length)
        self.boundary_single(CAT1)
        self.CAT1_pos = np.tile(CAT1,(self.n,1))
        
        CAT2 = np.arange(2+self.shifted_distance,2+self.tot_length+self.shifted_distance,self.repeated_length)
        self.boundary_single(CAT2)
        self.CAT2_pos = np.tile(CAT2,(self.n,1))
        
    def boundary_single(self,x):
        """Apply periodic boundary conditions to a single array."""
        x[x<0] += self.tot_length
        x[x >= self.tot_length] -= self.tot_length
        
    def boundary(self):
        """Apply periodic boundary conditions to both particles."""
        self.boundary_single(self.x1)
        self.boundary_single(self.x2)
        
    def pbc(self,dx):
        """Apply periodic boundary conditions to distance calculations."""
        dx[dx >= self.tot_length/2] -= self.tot_length
        dx[dx <= -self.tot_length/2] += self.tot_length
        
    def compute_dr(self):
        """Compute the minimal distance between the two particles."""
        self.dr = self.x2-self.x1
        self.pbc(self.dr)
        
    def compute_dCAT(self, x, CAT_pos):
        """
        Compute distance between particle positions and catalytic sites.
        
        Args:
            x (np.ndarray): Particle positions, shape (n,)
            CAT_pos (np.ndarray): Catalytic site positions, shape (n, num_motifs)
            
        Returns:
            np.ndarray: Distances to catalytic sites, shape (n, num_motifs)
            
        Notes:
            - Uses pre-allocated arrays (self.tiled_x1, self.tiled_x2) for efficiency
            - Applies periodic boundary conditions via self.pbc()
            - Optimized for repeated calls with same particle (x1 or x2)
            - Returns minimal distances (PBC-handled)
            - CAT positions: track 1 at 2, 2+repeated_length, etc.
            - CAT positions: track 2 at 2+shifted_distance, 2+shifted_distance+repeated_length, etc.
        """
        # Use pre-allocated arrays to avoid memory allocation overhead
        if x is self.x1:
            tiled_x = self.tiled_x1
        elif x is self.x2:
            tiled_x = self.tiled_x2
        else:
            # Fallback for other cases (shouldn't happen in normal usage)
            tiled_x = np.tile(x.reshape(-1,1),(1,self.num_motifs))
            dx_CAT = tiled_x-CAT_pos
            self.pbc(dx_CAT)
            return dx_CAT
        
        # Efficiently update the pre-allocated array
        tiled_x[:] = x.reshape(-1,1)
        dx_CAT = tiled_x-CAT_pos
        self.pbc(dx_CAT)
        return dx_CAT
    
    def compute_shifted_x_and_elementary_x(self):
        """Compute shifted and elementary coordinates for both particles."""
        # Use pre-allocated arrays to avoid memory allocation overhead
        # Update shifted coordinates in-place
        self.shifted_x1[:] = self.x1 + self.well_width/2
        self.boundary_single(self.shifted_x1)
        self.shifted_x2[:] = self.x2 + self.well_width/2 - self.shifted_distance
        self.boundary_single(self.shifted_x2)
        
        # Update elementary coordinates in-place (more efficient than modulo)
        self.elementary_shifted_x1[:] = self.shifted_x1 % self.repeated_length - self.well_width/2
        self.elementary_shifted_x2[:] = self.shifted_x2 % self.repeated_length - self.well_width/2

    # =============================================================================
    # MONTE CARLO SIMULATION FUNCTIONS
    # =============================================================================
    
    def Monte_Carlo_single_particle_poisson(self, x, CAT_pos, choices):
        """
        Alternative Monte Carlo method using Poisson process simulation.
        This allows for multiple events per time step with proper statistics.
        Vectorized for speed with preallocated arrays for performance.
        """
        # The distance between the CAT and the current ring position. 
        distance_x_CAT = np.abs(self.compute_dCAT(x,CAT_pos))
        k_attach = self.k_attach_r(distance_x_CAT)
        k_detach = self.k_detach_r(distance_x_CAT)
        
        # Calculate total rates for each particle
        total_time = self.dt * self.MC_steps
        
        # Use preallocated masks for eligible sites
        self.unblocked_mask[:] = (choices == 0)  # Sites that can be blocked
        self.blocked_mask[:] = (choices == 1)    # Sites that can be unblocked
        
        # Check for high rates (threshold: 0.01*100*0.005 = 0.005)
        threshold = 0.01
        
        # Only generate Poisson numbers for eligible sites
        # For attachment events (unblocked sites)
        if np.any(self.unblocked_mask):
            # Get rates only for unblocked sites (zero out blocked sites)
            np.copyto(self.k_attach_eligible, k_attach)
            self.k_attach_eligible[~self.unblocked_mask] = 0
            
            # Check for high attachment rates
            high_attach_rates = self.k_attach_eligible*total_time > threshold
            if np.any(high_attach_rates):
                print("WARNING: High attachment rates detected!")
                print("Max k_attach_eligible: {:.6f}".format(np.max(self.k_attach_eligible*total_time)))
                print("Threshold: {:.6f}".format(threshold))
                print("Number of high rates: {}".format(np.sum(high_attach_rates)))
            
            # Generate Poisson numbers for all sites (zeros for blocked sites)
            self.n_attach[:] = np.random.poisson(self.k_attach_eligible * total_time)
            # Apply the flips (unblocked -> blocked)
            self.flip_mask[:] = np.logical_and(self.n_attach > 0, self.unblocked_mask)
            choices[self.flip_mask] = 1
        
        # For detachment events (blocked sites)
        if np.any(self.blocked_mask):
            # Get rates only for blocked sites (zero out unblocked sites)
            np.copyto(self.k_detach_eligible, k_detach)
            self.k_detach_eligible[~self.blocked_mask] = 0
            
            # Check for high detachment rates
            high_detach_rates = self.k_detach_eligible*total_time > threshold
            if np.any(high_detach_rates):
                print("WARNING: High detachment rates detected!")
                print("Max k_detach_eligible: {:.6f}".format(np.max(self.k_detach_eligible*total_time)))
                print("Threshold: {:.6f}".format(threshold))
                print("Number of high rates: {}".format(np.sum(high_detach_rates)))
            
            # Generate Poisson numbers for all sites (zeros for unblocked sites)
            self.n_detach[:] = np.random.poisson(self.k_detach_eligible * total_time)
            # Apply the flips (blocked -> unblocked)
            self.flip_mask[:] = np.logical_and(self.n_detach > 0, self.blocked_mask)
            choices[self.flip_mask] = 0
        return choices
    
    def Monte_Carlo_step(self):
        """
        Perform Monte Carlo step for both particles.
        
        This method simulates attachment/detachment events at catalytic sites
        using Poisson process simulation for both particles.
        
        Updates:
            self.choices_1 (np.ndarray): Occupancy array for particle 1, shape (n, num_motifs)
            self.choices_2 (np.ndarray): Occupancy array for particle 2, shape (n, num_motifs)
            
        Notes:
            - Uses Poisson process simulation for multiple events per time step
            - Calculates attachment/detachment rates based on particle positions
            - Updates occupancy arrays (0=unoccupied, 1=occupied)
            - Called every MC_steps simulation steps
        """
        self.Monte_Carlo_single_particle_poisson(self.x1, self.CAT1_pos, self.choices_1)
        self.Monte_Carlo_single_particle_poisson(self.x2, self.CAT2_pos, self.choices_2)

    # =============================================================================
    # FORCE AND POTENTIAL CALCULATIONS
    # =============================================================================
    
    def coupling_force(self):
        """Calculate coupling force between particles."""
        return force_spring_(self.dr,self.k,self.x0,self.cross_distance)

    def LJ_force(self, dx_CAT, choices):
        """
        Calculate Lennard-Jones force from barriers using optimized sparse computation.
        
        Args:
            dx_CAT (np.ndarray): Distances to catalytic sites, shape (n, num_motifs)
            choices (np.ndarray): Occupancy array, shape (n, num_motifs), 1=occupied, 0=unoccupied
            
        Returns:
            np.ndarray: Total LJ force for each particle, shape (n,)
            
        Notes:
            - Uses pre-computed constants for efficiency
            - Only computes forces for occupied sites (choices == 1)
            - Significantly faster than computing all sites then masking
            - Requires self.LJ_const1 and self.LJ_const2 to be initialized
        """
        return force_LJ_optimized_sparse_precomputed(self.LJ_const1, self.LJ_const2, dx_CAT, choices)
    
    def force_calculation(self):
        """
        Calculate all forces acting on both particles.
        
        This method computes the total force on each particle from:
        1. Bare track potential (sinusoidal wells)
        2. Coupling force between particles (spring)
        3. Lennard-Jones forces from occupied catalytic sites
        
        Updates:
            self.f1 (np.ndarray): Total force on particle 1, shape (n,)
            self.f2 (np.ndarray): Total force on particle 2, shape (n,)
            
        Notes:
            - Calls compute_dr() to update particle separation
            - Calls compute_shifted_x_and_elementary_x() for coordinate transformations
            - Computes distances to catalytic sites for LJ forces
            - Applies coupling force with opposite signs to both particles
            - Checks for NaN values and exits if found
        """
        # preparation
        self.compute_dr()
        self.compute_shifted_x_and_elementary_x()
        dx_CAT1 = self.compute_dCAT(self.x1,self.CAT1_pos)
        dx_CAT2 = self.compute_dCAT(self.x2,self.CAT2_pos)
        
        # bare-track force
        self.f1 = force_bare_track_(self.elementary_shifted_x1,self.well_width,self.barrier_height)
        self.f2 = force_bare_track_(self.elementary_shifted_x2,self.well_width,self.barrier_height)
        
        # coupling force
        coupling_f = self.coupling_force()
        self.f1 += coupling_f
        self.f2 -= coupling_f
        
        # force from barriers
        self.f1 += self.LJ_force(dx_CAT1,self.choices_1)
        self.f2 += self.LJ_force(dx_CAT2,self.choices_2)
        self.check_nan(self.f1,"self.f1")
        self.check_nan(self.f2,"self.f2")

    def potential_bare_track(self,x):
        """Calculate bare track potential (for testing purposes)."""
        return potential_bare_track_(x,self.well_width,self.barrier_height)
    
    def potential_LJ(self,dx_CAT,choices):
        """Calculate Lennard-Jones potential from barriers."""
        #   from barriers
        U_all_barriers = potential_LJ_(self.epA, self.epR, dx_CAT, self.sigma)
        U_LJ_sum = np.sum(U_all_barriers*choices,axis = 1)
        return U_LJ_sum
    
    def potential_coupling(self,dr = None):
        """Calculate coupling potential."""
        if dr is None:
            dr = self.dr
        return potential_spring_(dr,self.k,self.x0,self.cross_distance)

    def potential_calculation(self):
        """Calculate total potential energy of the system."""
        self.compute_dr()
        self.compute_shifted_x_and_elementary_x()
        dx_CAT1 = self.compute_dCAT(self.x1,self.CAT1_pos)
        dx_CAT2 = self.compute_dCAT(self.x2,self.CAT2_pos)
        U_bare_1 = self.potential_bare_track(self.elementary_shifted_x1)
        U_bare_2 = self.potential_bare_track(self.elementary_shifted_x2)
        U_LJ_1 = self.potential_LJ(dx_CAT1,self.choices_1)
        U_LJ_2 = self.potential_LJ(dx_CAT2,self.choices_2)
        U_coupling = self.potential_coupling()
        U_tot = U_bare_1+U_bare_2+U_LJ_1+U_LJ_2+U_coupling
        return U_tot

    # =============================================================================
    # DYNAMICS INTEGRATION
    # =============================================================================
    
    def underdamped(self, rand1, rand2):
        """
        VRORV integrator for Langevin dynamics (matches C++ implementation).
        
        Args:
            rand1 (np.ndarray): Random numbers for particle 1 thermalization, shape (n,)
            rand2 (np.ndarray): Random numbers for particle 2 thermalization, shape (n,)
            
        Algorithm:
            V: Velocity update (quarter time step)
            R: Position update (half time step) 
            O: Ornstein-Uhlenbeck process (thermalization)
            R: Position update (half time step)
            V: Velocity update (quarter time step)
            
        Notes:
            - Matches C++ implementation exactly
            - Uses pre-computed constants for efficiency
            - Applies boundary conditions after position updates
            - Calls force_calculation() to update forces
            - Updates both particles (x1, p1) and (x2, p2)
        """
        
        # VRORV integration for particle 1 (vectorized)
        # V step: quarter time step velocity update
        self.p1 += 0.5 * self.dt * self.f1
        
        # R step: half time step position update
        self.x1 += 0.5 * self.p1 * self.dt_m
        
        # O step: Ornstein-Uhlenbeck process (thermalization)
        self.p1 = self.p1 * self.exp_gamma_dt + rand1 * self.sigma_thermal
        
        # R step: full time step position update
        self.x1 += 0.5 * self.p1 * self.dt_m
        
        # VRORV integration for particle 2 (vectorized)
        # V step: quarter time step velocity update
        self.p2 += 0.5 * self.dt * self.f2
        
        # R step: half time step position update
        self.x2 += 0.5 * self.p2 * self.dt_m
        
        # O step: Ornstein-Uhlenbeck process (thermalization)
        self.p2 = self.p2 * self.exp_gamma_dt + rand2 * self.sigma_thermal
        
        # R step: full time step position update
        self.x2 += 0.5 * self.p2 * self.dt_m
        
        # Apply boundary conditions
        self.boundary()
        
        # Check for large increments
        if np.any(np.abs(self.p1 * self.dt_m) > 1):
            print("increment too large for x1!")
            print(np.abs(self.p1 * self.dt_m).max())
        if np.any(np.abs(self.p2 * self.dt_m) > 1):
            print("increment too large for x2!")
            print(np.abs(self.p2 * self.dt_m).max())
        
        # Calculate new forces
        self.force_calculation()
        
        # Final V step: full time step velocity update
        self.p1 += 0.5 * self.dt * self.f1
        self.p2 += 0.5 * self.dt * self.f2

    # =============================================================================
    # STATE ANALYSIS AND TRACKING
    # =============================================================================
    
    def compute_integer_x1_x2(self):
        """Compute integer positions for state analysis using preallocated arrays."""
        self.x1_int[:] = np.round(self.x1).astype(np.int32)
        self.x1_int[self.x1_int == self.tot_length] = 0
        self.x2_int[:] = np.round(self.x2).astype(np.int32)
        self.x2_int[self.x2_int == self.tot_length] = 0
        
    def compute_new_x1_x2_core(self):
        """Compute core state positions for both particles."""
        self.compute_integer_x1_x2()
        x1_state = np.searchsorted(self.BIND1+self.repeated_length/2,self.x1_int) * self.repeated_length
        x2_state = np.searchsorted(self.BIND2+self.repeated_length/2,self.x2_int) * self.repeated_length
        x1_state[x1_state == self.tot_length] = 0
        x2_state[x2_state == self.tot_length] = 0
        return x1_state, x2_state
    
    def coarse_graining_states(self, step, option, transition_folder_name, transition_file_handles, idx_traj):
        """
        Coarse-grain states and track transitions with improved efficiency.
        
        Args:
            step: Current simulation step
            option: Operation mode (0=initialize, 1=update, 2=cleanup)
            transition_folder_name: Directory for transition files
            transition_file_handles: List of file handles
            idx_traj: Trajectory index
        """
        if option == 0:
            # Initialize state tracking
            self._initialize_state_tracking(step, transition_folder_name, transition_file_handles, idx_traj)
        elif option == 1:
            # Update states and detect transitions
            self._update_state_tracking(step, transition_folder_name, transition_file_handles, idx_traj)
        elif option == 2:
            # Cleanup file handles
            self._cleanup_file_handles(transition_file_handles)
    
    def _initialize_state_tracking(self, step, transition_folder_name, transition_file_handles, idx_traj):
        """Initialize state tracking variables and file handles."""
        # Compute initial core states
        self.x1_state_old, self.x2_state_old = self.compute_new_x1_x2_core()
        
        # Create core state arrays using preallocated arrays
        self.core_old[:, 0] = self.x1_state_old
        self.core_old[:, 1:] = self.choices_1
        
        # Initialize waiting time array (already preallocated)
        self.wt_array.fill(self.dt)
        
        # Initialize file handles
        self.recording_transitions(True, step, transition_folder_name, transition_file_handles, idx_traj, [])
    
    def _update_state_tracking(self, step, transition_folder_name, transition_file_handles, idx_traj):
        """Update state tracking and detect transitions with optimized efficiency."""
        # Compute new core states (this also updates self.x1_int and self.x2_int)
        x1_state_new, x2_state_new = self.compute_new_x1_x2_core()
        
        # Check if particles are in core positions using preallocated arrays
        self.x1_in_core.fill(False)  # Reset to False
        self.x2_in_core.fill(False)  # Reset to False
        
        # Determine in-core using PBC distances to nearest BIND positions within self.core_size
        # Use pre-allocated tiled arrays to avoid allocations
        self.tiled_x1[:] = self.x1.reshape(-1,1)
        dx_bind1 = self.tiled_x1 - self.BIND1_pos
        self.pbc(dx_bind1)
        self.x1_in_core[:] = np.any(np.abs(dx_bind1) <= 0.5*self.core_size, axis=1)
        
        self.tiled_x2[:] = self.x2.reshape(-1,1)
        dx_bind2 = self.tiled_x2 - self.BIND2_pos
        self.pbc(dx_bind2)
        self.x2_in_core[:] = np.any(np.abs(dx_bind2) <= 0.5*self.core_size, axis=1)
        
        # Update states only where needed (avoid unnecessary array creation)
        self.x1_state_new[:] = np.where(self.x1_in_core, x1_state_new, self.x1_state_old)
        self.x2_state_new[:] = np.where(self.x2_in_core, x2_state_new, self.x2_state_old)
        
        # Create new core state array using preallocated array
        self.core_new[:, 0] = self.x1_state_new
        self.core_new[:, 1:] = self.choices_1
        
        # Detect state changes efficiently using preallocated array
        self.state_changes[:] = np.any(self.core_new != self.core_old, axis=1)
        
        # Only process if there are changes
        if np.any(self.state_changes):
            flag_change = np.where(self.state_changes)[0]
            
            # Update waiting times efficiently (include this step)
            self.wt_array += self.dt
            
            # Record transitions only for changed particles using pre-reset waiting times
            self.recording_transitions(False, step, transition_folder_name, transition_file_handles, idx_traj, flag_change)
            
            # Reset waiting times for changed particles after recording
            self.wt_array[flag_change] = self.dt
            
            # Detect and count cycles efficiently
            self._detect_cycles()
        else:
            # No changes - just update waiting times
            self.wt_array += self.dt
        
        # Update old states for next iteration (use views when possible)
        self.x1_state_old[:] = self.x1_state_new
        self.x2_state_old[:] = self.x2_state_new
        self.core_old[:] = self.core_new
    
    def _detect_cycles(self):
        """Efficiently detect and count cycles for both particles using precomputed constants."""
        # Calculate state differences once
        dx1 = self.x1_state_new - self.x1_state_old
        dx2 = self.x2_state_new - self.x2_state_old
        
        # Use precomputed constants for cycle detection
        # Right cycles: either normal jump or wrap-around
        right_cycles_1 = (dx1 == self.right_jump) | (dx1 == self.right_wrap)
        right_cycles_2 = (dx2 == self.right_jump) | (dx2 == self.right_wrap)
        
        # Left cycles: either normal jump or wrap-around
        left_cycles_1 = (dx1 == self.left_jump) | (dx1 == self.left_wrap)
        left_cycles_2 = (dx2 == self.left_jump) | (dx2 == self.left_wrap)
        
        # Update cycle counters in-place
        self.right_cycles_x1[right_cycles_1] += 1
        self.left_cycles_x1[left_cycles_1] += 1
        self.right_cycles_x2[right_cycles_2] += 1
        self.left_cycles_x2[left_cycles_2] += 1
    
    def _cleanup_file_handles(self, transition_file_handles):
        """Close all transition file handles."""
        for file_handle in transition_file_handles:
            file_handle.close()

    def recording_transitions(self, init, step, transition_folder_name, transition_file_handles, idx_traj, flag_change):
        """
        Record state transitions to files with improved efficiency.
        
        Args:
            init: Whether to initialize file handles (True) or record data (False)
            step: Current simulation step
            transition_folder_name: Directory for transition files
            transition_file_handles: List of file handles
            idx_traj: Trajectory index
            flag_change: Indices of particles that changed state
        """
        if init:
            # Initialize file handles for all replicas
            self._initialize_transition_files(transition_folder_name, transition_file_handles, idx_traj)
        else:
            # Record transition data for particles that changed state
            self._record_transition_data(step, transition_file_handles, flag_change)
    
    def _initialize_transition_files(self, transition_folder_name, transition_file_handles, idx_traj):
        """Initialize file handles for transition recording."""
        for i in range(self.n):
            filename = f"{transition_folder_name}/{((idx_traj-1)*self.n+i+1):04d}.txt"
            file_handle = open(filename, "w")
            transition_file_handles.append(file_handle)
    
    def _record_transition_data(self, step, transition_file_handles, flag_change):
        """Record transition data for particles that changed state."""
        current_time = self.dt * np.float64(step)
        
        for particle_idx in flag_change:
            file_handle = transition_file_handles[particle_idx]
            
            # Prepare data for writing
            core_state_str = ' '.join(map(str, self.core_old[particle_idx]))
            
            # Write transition data using f-string for better performance
            transition_line = (f"{current_time}\t"
                             f"{self.wt_array[particle_idx]}\t"
                             f"{self.x1[particle_idx]}\t"
                             f"{core_state_str}\t"
                             f"{self.right_cycles_x1[particle_idx]}\t"
                             f"{self.left_cycles_x1[particle_idx]}\n")
            
            file_handle.write(transition_line)

    # =============================================================================
    # SIMULATION EXECUTION
    # =============================================================================
    
    def propagation_underdamped_diffusion_jump_motor(self, steps, idx_traj):
        """Main simulation propagation function."""
        print("parallel job ID = "+str(idx_traj))
        print("Launch jump-diffsuion simulations...")
        folder_name = "simulation_data"
        transition_folder_name = "transition_data"
        file_handles = []
        transition_file_handles=[]
        time.sleep(3)
        for i in range(self.n):
            filename = str(folder_name)+"/%04d.txt" %((idx_traj-1)*self.n+i+1)
            file_handle = open(filename, "w")  # Open file in write mode
            file_handles.append(file_handle)   # Store file handle in a list
        self.coarse_graining_states(0,0,transition_folder_name,transition_file_handles,idx_traj)
        #self.checkpoint(True)
        self.force_calculation() 
        Nprint = np.int32(steps/10)
        counts = 0
        Nlag = 10000 # the frequency to write the configurations into the pool
        #c2 = 0
        ############debug############
        for i in range(steps):
            c1 = i%self.cycle
            if c1 == 0:
                #random numbers (with chunks), otherwise the cluster cannot run it.
                self.rand1[:] = np.random.normal(0, 1, (self.cycle, self.n))  # Reuse preallocated array
                self.rand2[:] = np.random.normal(0, 1, (self.cycle, self.n))  # Reuse preallocated array
            self.underdamped(self.rand1[c1],self.rand2[c1])
            self.check_nan(self.x1,"self.x1")
            self.check_nan(self.x2,"self.x2")
            if (i+1)%self.MC_steps == 0:
                # c2 = np.int32(i/self.MC_steps)%(self.cycle)
                # if c2 == 0:
                #     self.MC_rand1 = np.random.uniform(0,1,(self.cycle,self.n, self.num_motifs));
                #     self.MC_rand2 = np.random.uniform(0,1,(self.cycle,self.n, self.num_motifs));
                self.Monte_Carlo_step()
            #self.checkpoint(False)
            self.coarse_graining_states(i,1,transition_folder_name,transition_file_handles,idx_traj)
            if (i+1)%Nprint == 0:
                print("Finished: "+str(np.int32((i+1)/Nprint * 10))+"%")
                sys.stdout.flush()
                for file_handle in file_handles:
                    file_handle.flush() 
                for transition_file_handle in transition_file_handles:
                    transition_file_handle.flush()
                              
            if (i)%Nlag == 0 and i >= Nlag:
                print("step: "+str(i))
                counts += 1
                # Use preallocated trajectory array for better performance
                current_time = i * self.dt
                self.traj_array[0] = current_time
                self.traj_array[1:self.n+1] = self.x1
                self.traj_array[self.n+1:2*self.n+1] = self.x2
                self.traj_array[2*self.n+1:3*self.n+1] = self.p1
                self.traj_array[3*self.n+1:4*self.n+1] = self.p2
                self.traj_array[4*self.n+1:5*self.n+1] = self.right_cycles_x1
                self.traj_array[5*self.n+1:6*self.n+1] = self.left_cycles_x1
                self.traj_array[6*self.n+1:7*self.n+1] = self.right_cycles_x2
                self.traj_array[7*self.n+1:8*self.n+1] = self.left_cycles_x2
                self.traj_array[8*self.n+1:8*self.n+1+self.n*self.num_motifs] = np.transpose(self.choices_1).flatten()
                self.traj_array[8*self.n+1+self.n*self.num_motifs:] = np.transpose(self.choices_2).flatten()
                traj = self.traj_array
                for k,file_handle in enumerate(file_handles):
                    single_traj = traj[[0,k+1,self.n+k+1,self.n*2+k+1,self.n*3+k+1,self.n*4+k+1,self.n*5+k+1]]
                    array_string = ' '.join(map(str, single_traj))
                    file_handle.write(array_string)
                    file_handle.write('\t')
                    # record the blocking groups
                    for u in range(self.num_motifs*2):
                        file_handle.write(str(traj[self.n*(6+u)+k+1]))
                        file_handle.write('\t')
                    file_handle.write('\n')
                    #file_handle.flush()                     
        for file_handle in file_handles:
            file_handle.close()
        self.coarse_graining_states(steps,2,transition_folder_name,transition_file_handles,idx_traj)
        ###########compute other quantities############
        # left and right hopping rates:
        if steps > Nlag*2:
            right_cyc_rate_mean,right_cyc_rate_err,left_cyc_rate_mean,left_cyc_rate_err = hopping_rates(self.n)
            np.savetxt("hopping_rates.txt",np.array([right_cyc_rate_mean,right_cyc_rate_err,left_cyc_rate_mean,left_cyc_rate_err]))
        
    def parallel_propagation_underdamped_diffusion_jump_motor(self,idx_traj,steps = 1000000):
        """Initialize and run parallel simulation."""
        folder_name = "simulation_data"
        if not os.path.exists(folder_name) and idx_traj ==1:
            os.makedirs(folder_name)
        folder_name = "transition_data"
        if not os.path.exists(folder_name) and idx_traj ==1:
            os.makedirs(folder_name)
        self.initialize_system()
        self.propagation_underdamped_diffusion_jump_motor(steps,idx_traj)

# =============================================================================
# HIGH-LEVEL INTERFACE CLASS
# =============================================================================

class DiffusionJumpMotor(diffusion_jump_motor):
    """
    High-level interface for the Diffusion-Jump Motor simulation.
    Provides a cleaner API for parameter setting and simulation execution.
    """
    
    def __init__(self,
                 # equilibrium
                 MC_steps = 100,
                 # potential parameters
                 barrier_height=3.15, well_width=6, repeated_length=12, num_motifs=4,
                 # interaction parameters
                 epR=1e5, 
                 # rate constant parameters
                 k_attach_far=2e-4, center_attach=4.6, f=0.001, k_detach_far=1.5e-4, eta=0,
                 # coupling parameters
                 shifted_distance=0, coupling_strength=0, coupling_center=0,gamma = 6,
                 # core detection parameter
                 core_size=1.0):
        """
        Initialize the diffusion jump motor with all required parameters.
        """
        # Call the parent class constructor
        super().__init__()

        # Initialize potential parameters
        self.MC_steps = MC_steps
        self.barrier_height = barrier_height
        self.well_width = well_width
        self.repeated_length = repeated_length
        self.num_motifs = num_motifs

        # Initialize interaction parameters
        self.epR = epR
        self.k_attach_far = k_attach_far
        self.center_attach = center_attach
        self.spread_attach = spread_attach
        self.k_detach_far = k_detach_far
        self.eta = eta

        # Initialize coupling parameters
        self.shifted_distance = shifted_distance
        self.k = coupling_strength
        self.x0 = coupling_center
        # Motor simulation parameters
        self.m = 12  # ring mass
        self.gamma = gamma #0.5 * self.m
        self.n = 100
        self.dt = 0.005
        self.beta = 2
        self.cross_distance = 10
        
        # core detection parameter
        self.core_size = core_size

    def run_simulation(self, trialID, steps):
        """
        Run the motor simulation with the provided trial ID and steps.
        """
        print(f"Running simulation with trial ID: {trialID} and steps: {steps}")
        super().parallel_propagation_underdamped_diffusion_jump_motor(trialID, steps)
        
    def parallel_run_simulation(self,parallel_jobs,steps):
        """Run multiple parallel simulations."""
        idx_list = np.array([i for i in range(0, parallel_jobs, 1)])
        Parallel(n_jobs=int(parallel_jobs))(delayed(self.run_simulation)(trialID+1,steps) for trialID in idx_list)
        
# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    motor = diffusion_jump_motor()
    motor.k = 0
    motor.x0 = 0
    motor.shifted_distance = 0
    motor.k_attach_far = 0.0002
    steps = 1000000
    motor.MC_steps = 100
    motor.eta = 0
    
    motor.parallel_propagation_underdamped_diffusion_jump_motor(idx_traj=1, steps=steps)

