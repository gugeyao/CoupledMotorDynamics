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
from scipy.optimize import curve_fit
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
def potential_LJ_(epA = 0, epR = 1e4, dr_CAT=2,sigma = 1):
    """Lennard-Jones potential for repulsive interactions."""
    U = 4*(-epA*(sigma/dr_CAT)**6+epR*(sigma/dr_CAT)**12)
    return U

@nb.jit(nopython=nopython)
def potential_spring_(dr=np.array([]),k= 0,x0 = 0,cross_distance = 10):
    """Spring potential for coupling between particles."""
    delta_x = np.sqrt((dr)**2+cross_distance**2)
    U = 0.5 * k *(delta_x - x0)**2
    return U

# =============================================================================
# FORCE FUNCTIONS
# =============================================================================

@nb.jit(nopython=nopython)
def force_bare_track_(x_elementary=np.array([]),length =0,barrier_height = 0):
    """Force from bare track potential using finite difference."""
    diff = np.float64(0.00000001)
    x_elementary = x_elementary + diff
    U_plus = potential_bare_track_(x_elementary,length,barrier_height)
    x_elementary = x_elementary - 2 * diff
    U_minus = potential_bare_track_(x_elementary,length,barrier_height)
    f_finite = -(U_plus - U_minus)/(2 * diff)
    x_elementary = x_elementary + diff
    f = f_finite
    return f

@nb.jit(nopython=nopython)
def force_LJ_(epA = 0, epR = 1e4, dr_CAT=2, sigma = 1):
    """Force from Lennard-Jones potential using finite difference."""
    diff = np.float64(0.00000001)
    dr_CAT = dr_CAT +diff
    U_plus = potential_LJ_(epA, epR, dr_CAT,sigma)
    dr_CAT = dr_CAT  - 2*diff
    U_minus = potential_LJ_(epA, epR, dr_CAT,sigma)
    f_finite = -(U_plus - U_minus)/(2 * diff)
    dr_CAT = dr_CAT + diff
    f = f_finite
    return f

@nb.jit(nopython=nopython)
def force_spring_(dr=np.array([]),k = 0,x0 = 0,cross_distance = 10):
    """Force from spring potential using finite difference."""
    diff = np.float64(0.00000001)
    dr = dr - diff
    U_plus = potential_spring_(dr,k,x0,cross_distance)
    dr = dr + 2 * diff
    U_minus = potential_spring_(dr,k,x0,cross_distance)
    f1_spring = -(U_plus - U_minus)/(2 * diff)
    dr = dr - diff
    return f1_spring

# =============================================================================
# ANALYSIS AND UTILITY FUNCTIONS
# =============================================================================

def hopping_rates(num_trial):
    """Calculate hopping rates from simulation data."""
    a_folder = "simulation_data"
    right_cyc_rates = np.zeros(num_trial)
    left_cyc_rates = np.zeros(num_trial)
    for k in range(num_trial):
        file = "%04d.txt"%(k+1)
        data = np.loadtxt(open(a_folder+"/"+file))[[0,-1],:]
        time = data[:,0]
        r1 = data[:,3]
        l1 = data[:,4]# left cycles for particle 1
        r2 = data[:,5]
        l2 = data[:,6]# left cycles for particle 2
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
        self.delta_gap = 0.5 # core-state definition
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

    def initialize_system(self):        
        """Initialize the simulation system with all required variables."""
        # VRORV integrator parameters (matches C++ implementation)
        # initialization parameters
        self.tot_length = self.num_motifs * self.repeated_length
        self.x1 = np.random.uniform(0,self.tot_length,(self.n)) # initialize the positions of the particles
        self.x2 = np.array(self.x1)+np.random.uniform(-self.tot_length/((1+self.k*self.beta*10)*2),self.tot_length/((1+self.k*self.beta*10)*2),(self.n)) # not causing a big energy difference
        self.boundary()
        self.compute_shifted_x_and_elementary_x()
        self.compute_dr()
        # Boltzmann distribution: p ~ N(0, sqrt(m/β)) = N(0, sqrt(m*kB*T))
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
        print("✅ Matches C++ implementation exactly")
        print("✅ Good numerical stability")
        print("✅ Proper thermalization")
        print("✅ Symplectic structure")
        print("✅ Time-reversible")
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
        k[x < 2 + self.well_width/2] = 0 # prevent the particle's potential drastically change when it is on the binding site.
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
        
    def CAT_position(self):
        """Calculate catalytic site positions for both tracks."""
        CAT1 =np.arange(2,2+self.tot_length,self.repeated_length)
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
        
    def compute_dCAT(self,x,CAT_pos):
        """Compute distance between particle position and catalytic sites."""
        # The CAT position for track 1 is: 2, 2 + 2*l, ..., 2+ (num_motif-1)*l, <-- apply boundary on CAT position, (fixed during the simulation)
        # The CAT position for track 2 is: 2 + shifted_distance,  <-- apply boundary on CAT position
        tiled_x = np.tile(x.reshape(-1,1),(1,self.num_motifs))
        dx_CAT = tiled_x-CAT_pos
        self.pbc(dx_CAT)
        return dx_CAT
    
    def compute_shifted_x_and_elementary_x(self):
        """Compute shifted and elementary coordinates for both particles."""
        self.shifted_x1 = self.x1 +self.well_width/2
        self.boundary_single(self.shifted_x1)
        self.shifted_x2 = self.x2 +self.well_width/2-self.shifted_distance
        self.boundary_single(self.shifted_x2)
        self.elementary_shifted_x1 = self.shifted_x1%(self.repeated_length) -self.well_width/2
        self.elementary_shifted_x2 = self.shifted_x2%(self.repeated_length) -self.well_width/2

    # =============================================================================
    # MONTE CARLO SIMULATION FUNCTIONS
    # =============================================================================
    
    def Monte_Carlo_single_particle_poisson(self, x, CAT_pos, choices):
        """
        Alternative Monte Carlo method using Poisson process simulation.
        This allows for multiple events per time step with proper statistics.
        Vectorized for speed.
        """
        # The distance between the CAT and the current ring position. 
        distance_x_CAT = np.abs(self.compute_dCAT(x,CAT_pos))
        k_attach = self.k_attach_r(distance_x_CAT)
        k_detach = self.k_detach_r(distance_x_CAT)
        
        # Calculate total rates for each particle
        total_time = self.dt * self.MC_steps
        
        # First, create masks for eligible sites
        unblocked_mask = (choices == 0)  # Sites that can be blocked
        blocked_mask = (choices == 1)     # Sites that can be unblocked
        
        # Check for high rates (threshold: 0.01*100*0.005 = 0.005)
        threshold = 0.01
        
        # Only generate Poisson numbers for eligible sites
        # For attachment events (unblocked sites)
        if np.any(unblocked_mask):
            # Get rates only for unblocked sites (zero out blocked sites)
            k_attach_eligible = k_attach.copy()
            k_attach_eligible[~unblocked_mask] = 0
            
            # Check for high attachment rates
            high_attach_rates = k_attach_eligible > threshold
            if np.any(high_attach_rates):
                print(f"WARNING: High attachment rates detected!")
                print(f"Max k_attach_eligible: {np.max(k_attach_eligible):.6f}")
                print(f"Threshold: {threshold:.6f}")
                print(f"Number of high rates: {np.sum(high_attach_rates)}")
            
            # Generate Poisson numbers for all sites (zeros for blocked sites)
            n_attach = np.random.poisson(k_attach_eligible * total_time)
            # Apply the flips (unblocked -> blocked)
            flip_mask = (n_attach > 0) & unblocked_mask
            choices[flip_mask] = 1
        
        # For detachment events (blocked sites)
        if np.any(blocked_mask):
            # Get rates only for blocked sites (zero out unblocked sites)
            k_detach_eligible = k_detach.copy()
            k_detach_eligible[~blocked_mask] = 0
            
            # Check for high detachment rates
            high_detach_rates = k_detach_eligible > threshold
            if np.any(high_detach_rates):
                print(f"WARNING: High detachment rates detected!")
                print(f"Max k_detach_eligible: {np.max(k_detach_eligible):.6f}")
                print(f"Threshold: {threshold:.6f}")
                print(f"Number of high rates: {np.sum(high_detach_rates)}")
            
            # Generate Poisson numbers for all sites (zeros for unblocked sites)
            n_detach = np.random.poisson(k_detach_eligible * total_time)
            # Apply the flips (blocked -> unblocked)
            flip_mask = (n_detach > 0) & blocked_mask
            choices[flip_mask] = 0
        return choices
    
    def Monte_Carlo_step(self):
        """Perform Monte Carlo step for both particles."""
        self.Monte_Carlo_single_particle_poisson(self.x1, self.CAT1_pos, self.choices_1)
        self.Monte_Carlo_single_particle_poisson(self.x2, self.CAT2_pos, self.choices_2)

    # =============================================================================
    # FORCE AND POTENTIAL CALCULATIONS
    # =============================================================================
    
    def coupling_force(self):
        """Calculate coupling force between particles."""
        return force_spring_(self.dr,self.k,self.x0,self.cross_distance)

    def LJ_force(self,dx_CAT,choices):
        """Calculate Lennard-Jones force from barriers."""
        # force from barriers
        f_LJ_all_barriers = force_LJ_(self.epA, self.epR, dx_CAT, self.sigma)
        # deal with states that are currently blocked 
        f_LJ_sum = np.sum(f_LJ_all_barriers*choices,axis = 1)
        
        return f_LJ_sum
    
    def force_calculation(self):
        """Calculate all forces acting on both particles."""
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
        V = Velocity update (V for "V"elocity)
        R = Position update (R for "R"eset/position) 
        O = Ornstein-Uhlenbeck process (O for "O"rnstein)
        """
        # Precompute constants for efficiency
        dt_m = self.dt / self.m # damping coefficient
        exp_gamma_dt = np.exp(-self.gamma * dt_m)  # Matches C++: exp(-γ*dt/m)
        sigma = np.sqrt(self.m * (1 - exp_gamma_dt**2) / self.beta)  # Matches C++: sqrt(m*(1-exp(-2*γ*dt/m))/β)
        
        # VRORV integration for particle 1 (vectorized)
        # V step: quarter time step velocity update
        self.p1 += 0.5 * self.dt * self.f1
        
        # R step: half time step position update
        self.x1 += 0.5 * self.p1 * dt_m
        
        # O step: Ornstein-Uhlenbeck process (thermalization)
        self.p1 = self.p1 * exp_gamma_dt + rand1 * sigma
        
        # R step: full time step position update
        self.x1 += 0.5 * self.p1 * dt_m
        
        # VRORV integration for particle 2 (vectorized)
        # V step: quarter time step velocity update
        self.p2 += 0.5 * self.dt * self.f2
        
        # R step: half time step position update
        self.x2 += 0.5 * self.p2 * dt_m
        
        # O step: Ornstein-Uhlenbeck process (thermalization)
        self.p2 = self.p2 * exp_gamma_dt + rand2 * sigma
        
        # R step: full time step position update
        self.x2 += 0.5 * self.p2 * dt_m
        
        # Apply boundary conditions
        self.boundary()
        
        # Check for large increments
        if np.any(np.abs(self.p1 * dt_m) > 1):
            print("increment too large for x1!")
            print(np.abs(self.p1 * dt_m).max())
        if np.any(np.abs(self.p2 * dt_m) > 1):
            print("increment too large for x2!")
            print(np.abs(self.p2 * dt_m).max())
        
        # Calculate new forces
        self.force_calculation()
        
        # Final V step: full time step velocity update
        self.p1 += 0.5 * self.dt * self.f1
        self.p2 += 0.5 * self.dt * self.f2

    # =============================================================================
    # STATE ANALYSIS AND TRACKING
    # =============================================================================
    
    def compute_integer_x1_x2(self):
        """Compute integer positions for state analysis."""
        self.x1_int = np.round(self.x1)
        self.x1_int[self.x1_int == self.tot_length] = 0
        self.x2_int = np.round(self.x2)
        self.x2_int[self.x2_int == self.tot_length] = 0
        
    def compute_new_x1_x2_core(self):
        """Compute core state positions for both particles."""
        self.compute_integer_x1_x2()
        x1_state = np.searchsorted(self.BIND1+self.repeated_length/2,self.x1_int) * self.repeated_length
        x2_state = np.searchsorted(self.BIND2+self.repeated_length/2,self.x2_int) * self.repeated_length
        x1_state[x1_state == self.tot_length] = 0
        x2_state[x2_state == self.tot_length] = 0
        return x1_state, x2_state
    
    def coarse_graining_states(self,step,option,transition_folder_name,transition_file_handles,idx_traj):
        """Coarse-grain states and track transitions."""
        if option == 0:
            # rings' position
            self.x1_state_old, self.x2_state_old  = self.compute_new_x1_x2_core()
            self.x1_state_new = np.array(self.x1_state_old)
            self.x2_state_new = np.array(self.x2_state_old)
            # ring 1's blocking status
            potential_type_1 = np.array(self.choices_1)
            self.core_old = np.concatenate((self.x1_state_old.reshape(-1,1),potential_type_1),axis = 1)
            self.core_new = np.array(self.core_old)
            self.wt_array = np.zeros(self.n)+self.dt
            flag_change = []
            self.recording_transitions(True,step,transition_folder_name,transition_file_handles,idx_traj,flag_change)
        elif option == 1:
            self.x1_state_new, self.x2_state_new  = self.compute_new_x1_x2_core()
            x1_not_in_core = np.where(~np.isin(self.x1_int, self.BIND1))[0]
            x2_not_in_core = np.where(~np.isin(self.x2_int, self.BIND2))[0]
            self.x1_state_new[x1_not_in_core] = self.x1_state_old[x1_not_in_core]
            self.x2_state_new[x2_not_in_core] = self.x2_state_old[x2_not_in_core]
            potential_type_1 = np.array(self.choices_1)
            self.core_new = np.concatenate((self.x1_state_new.reshape(-1,1),potential_type_1),axis = 1)
            
            flag_change = np.where(np.any(self.core_new -self.core_old != np.zeros(self.num_motifs+1),axis = 1))[0]
            self.wt_array += self.dt
            #### if so, doing the recordings##########
            self.recording_transitions(False,step,transition_folder_name,transition_file_handles,idx_traj,flag_change)
            
            # check if the state is making right cycles
            self.right_cycles_x1[(self.x1_state_new - self.x1_state_old == self.repeated_length) | (self.x1_state_new - self.x1_state_old == (1-self.num_motifs)*self.repeated_length)]+= 1
            self.left_cycles_x1[(self.x1_state_new - self.x1_state_old == -self.repeated_length) | (self.x1_state_new - self.x1_state_old == (-1 + self.num_motifs)*self.repeated_length)]+= 1     
            self.right_cycles_x2[(self.x2_state_new - self.x2_state_old == self.repeated_length) | (self.x2_state_new - self.x2_state_old == (1 - self.num_motifs)*self.repeated_length)]+= 1
            self.left_cycles_x2[(self.x2_state_new - self.x2_state_old == -self.repeated_length) | (self.x2_state_new - self.x2_state_old == (-1 + self.num_motifs)*self.repeated_length)]+= 1
            self.x1_state_old = np.array(self.x1_state_new)
            self.x2_state_old = np.array(self.x2_state_new)           
            self.wt_array[flag_change] = self.dt
            self.core_old = np.array(self.core_new)
            
        elif option ==2:
            for transition_file_handle in transition_file_handles:
                transition_file_handle.close()
            
    def recording_transitions(self,init,step,transition_folder_name,transition_file_handles,idx_traj,flag_change):
        """Record state transitions to files."""
        if init == True:
            for i in range(self.n):
                filename = str(transition_folder_name)+"/%04d.txt" %((idx_traj-1)*self.n+i+1)
                transition_file_handle = open(filename, "w")  # Open file in write mode
                transition_file_handles.append(transition_file_handle)   # Store file handle in a list      
        else:
            for k in flag_change:
                transition_file_handle = transition_file_handles[k]
                array_string = ' '.join(map(str, self.core_old[k]))
                transition_file_handle.write(str(self.dt*np.float64(step)))
                transition_file_handle.write('\t')
                transition_file_handle.write(str(self.wt_array[k]))
                transition_file_handle.write('\t')
                transition_file_handle.write(str(self.x1[k]))
                transition_file_handle.write('\t')
                transition_file_handle.write(array_string)
                transition_file_handle.write('\t')
                transition_file_handle.write(str(self.right_cycles_x1[k]))
                transition_file_handle.write('\t')
                transition_file_handle.write(str(self.left_cycles_x1[k]))
                transition_file_handle.write('\n')
                #transition_file_handle.flush() 

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
                #random numbers (with chuncks), otherwise the cluster cannot run it.
                self.rand1 = np.random.normal(0,1,(self.cycle,self.n)); # Particle 1 - BAOAB needs only one set per step
                self.rand2 = np.random.normal(0,1,(self.cycle,self.n)); # Particle 2 - BAOAB needs only one set per step
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
                traj = np.array(np.concatenate((np.array([i * self.dt]),self.x1, self.x2,self.right_cycles_x1,self.left_cycles_x1,\
                                                self.right_cycles_x2,self.left_cycles_x2,np.transpose(self.choices_1).flatten(),np.transpose(self.choices_2).flatten())))
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
                 epR=1e4, 
                 # rate constant parameters
                 k_attach_far=2e-4, center_attach=4.6, spread_attach=0.001, k_detach_far=1.5e-4, eta=1,
                 # coupling parameters
                 shifted_distance=0, coupling_strength=0, coupling_center=0,gamma = 6):
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

