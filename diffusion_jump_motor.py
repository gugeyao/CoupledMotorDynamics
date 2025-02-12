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
#have validated the potential function
# x is elementary x
@nb.jit(nopython=nopython)
def potential_bare_track_(x=np.array([]),length =0,barrier_height = 0):
    indices = np.where((x >= -length/2) & (x<= length/2))[0]
    U = np.zeros_like(x)+barrier_height
    U[indices] = barrier_height*np.sin(2*np.pi/length*(x[indices]-length/4))
    return U

@nb.jit(nopython=nopython)
def potential_LJ_(epA = 0, epR = 1e4, dr_CAT=2,sigma = 1):
    U = 4*(-epA*(sigma/dr_CAT)**6+epR*(sigma/dr_CAT)**12)
    return U

@nb.jit(nopython=nopython)
def potential_spring_(dr=np.array([]),k= 0,x0 = 0,cross_distance = 10):
    delta_x = np.sqrt((dr)**2+cross_distance**2)
    U = 0.5 * k *(delta_x - x0)**2
    return U

@nb.jit(nopython=nopython)
def potential_FENE_(dr=np.array([]),kNL = 0,n= 10,cross_distance = 10):
    delta_x = np.sqrt((dr)**2+cross_distance**2)
    U = -0.5 * kNL * delta_x**2 * np.log(1 - (delta_x/n)**2)
    return U

@nb.jit(nopython=nopython)
def force_bare_track_(x_elementary=np.array([]),length =0,barrier_height = 0):
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
    diff = np.float64(0.00000001)
    dr = dr - diff
    U_plus = potential_spring_(dr,k,x0,cross_distance)
    dr = dr + 2 * diff
    U_minus = potential_spring_(dr,k,x0,cross_distance)
    f1_spring = -(U_plus - U_minus)/(2 * diff)
    dr = dr - diff
    return f1_spring

@nb.jit(nopython=nopython)
def force_FENE_(dr=np.array([]),kNL = 0,n= 10,cross_distance = 10):
    diff = np.float64(0.00000001)
    dr = dr - diff
    U_plus = potential_FENE_(dr,kNL,n,cross_distance)
    dr = dr + 2 * diff
    U_minus = potential_FENE_(dr,kNL,n,cross_distance)
    f1_spring = -(U_plus - U_minus)/(2 * diff)
    dr = dr - diff
    return f1_spring

def hopping_rates(num_trial):
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

class diffusion_jump_motor(object):    
    def __init__(self):
        # potential parameters
        self.k = 0 # spring strength
        self.x0 = 0 # spring center
        self.coupling_type = "spring" # or FENE potential
        self.well_width = 4 # repeated length of the sin function
        self.shifted_distance = 0 # the shifted phase between the two rings
        self.barrier_height = 1.17 # the barrier height of the sinx well
        self.repeated_length = 12
        self.cross_distance = 10 # the distance between the double potential
        self.epA = 0
        self.epR = 1e4
        self.sigma = 1 # radii of the repulsion
        # system parameters
        self.beta = 2
        self.m = 1
        self.n = 100 #number of replicas (traj.)
        self.dt = 0.005 # to prevent too large step
        self.gamma = 1
        self.delta_gap = 0.5 # core-state definition
        self.cycle = 10000 # the length of generating random numbers
        # number of periodic wells
        self.num_motifs = 4
        
        ############ rate functions#############
        self.k_far = 1 # factor of e^{-beta E_{C-Ring}}
        self.k_close = 0
        self.k_attach_right_C = 3e-3
        self.k_attach_left_C = 3e-3
        self.mu_C_std = -3.915
        self.E_C_track = 5.414
        self.k_attach_right_FTC = 2e-5
        self.k_attach_left_FTC = 0 #1e-7 makes the simulation blowing up
        self.k_detach_right_FTC = 0
        self.k_detach_left_FTC = 0
        self.center_FTC = 4.6
        self.spread_FTC = 0.001 # as small as possible to avoid 
        self.conc_FTC = 10 # this is decided by chemical potential
        self.conc_ETC = 0
        self.conc_C = 0
        self.eta = 0.9 # factor for deciding the fraction of attachment and detachment rate change as a function of distance to the ring.
        self.MC_steps = 100
        self.MC_x1_on = True # if the MC move is turned on for x1. Suppose that Ring1's GCMC move can be turned off independent to x2.

    def initialize_system(self):        
        #underdamped integrator parameters
        self.expgamma=np.exp(-self.gamma*self.dt/2/self.m);
        self.stdx=np.sqrt(self.m*(1-self.expgamma**2)/self.beta);
        # initialization parameters
        self.tot_length = self.num_motifs * self.repeated_length
        self.x1 = np.random.uniform(0,self.tot_length,(self.n)) # initial positions are all set to zero, the trajectories will be used after some burning time
        if self.coupling_type == "spring":
            self.x2 = np.array(self.x1)+np.random.uniform(-self.tot_length/((1+self.k*self.beta*10)*2),self.tot_length/((1+self.k*self.beta*10)*2),(self.n)) # not causing a big energy difference
        else:
            self.x2 = np.array(self.x1)
        self.boundary()
        self.compute_shifted_x_and_elementary_x()
        self.compute_dr()
        self.p1 = np.random.normal(0,np.sqrt(1/self.m*self.beta),(self.n))
        self.p2 = np.random.normal(0,np.sqrt(1/self.m*self.beta),(self.n))
        self.f1 = np.zeros((self.n))
        self.f2 = np.zeros((self.n))
        self.rand1 = np.random.normal(0,self.stdx,(self.cycle,2,self.n));
        self.rand2 = np.random.normal(0,self.stdx,(self.cycle,2,self.n));
        
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
        
        # probabilities of jumping to the next state
        self.P1 = np.zeros((self.n,self.num_motifs)) # the probablity of changing each blocking groups
        self.P2 = np.zeros((self.n,self.num_motifs)) # the probablity of changing each blocking groups
        
        #  the site being blocked or unblocked
        self.choices_1 = -np.ones((self.n,self.num_motifs+1)) # each choice can be blocked or unblocked, -1: unblocked, 1:  blocked, the num_motif+1 the last one if for changing nothing. not in use
        self.choices_2 = -np.ones((self.n,self.num_motifs+1)) # each choice can be blocked or unblocked, -1: unblocked, 1:  blocked
        # random number of deciding the MC steps
        self.MC_rand1 = np.random.uniform(0,1,(self.cycle,self.n));
        self.MC_rand2 = np.random.uniform(0,1,(self.cycle,self.n));
        
        # fit fermi functions for C particle
        self.compute_prefactor()
        self.compute_k_left()
        self.Fit_fermi_function_for_expfactor()
        self.compute_k_detach_C()
        self.CAT_position()
        self.BIND_position()
        self.force_calculation()
        if self.x0 < self.cross_distance and self.coupling_type == "FENE":
            print("too small number of linker beads!")
            sys.exit()
        self.print_out_system_info()
        
    def print_out_system_info(self):
        ##############print info###############
        print("========System info=========")
        print("beta = "+str(self.beta))
        print("m = "+str(self.m))
        print("dt = "+str(self.dt))
        print("gamma = "+str(self.gamma))
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
        print("Ring 1 is in nonequilibrium or not: "+str(self.MC_x1_on))
        print("")
        
        print("========Coupling info=========")
        print("spring strength = "+str(self.k))
        print("spring center = "+str(self.x0))
        print("shifted distance = "+str(self.shifted_distance))
        print("cross distance = "+str(self.cross_distance))
        print("coupling type = "+str(self.coupling_type))
        print("")
        print("Parameters of rates:")
        print("k_far = "+str(self.k_far))
        print("k_close = "+str(self.k_close))
        print("A = "+str(self.prefactor))
        
        print("========C==========")
        print("k_attach_right_C = "+str(self.k_attach_right_C))
        print("k_attach_left_C = "+str(self.k_attach_left_C))
        print("center_C = "+str(self.center_C))
        print("spread_C = "+str(self.spread_C))
        
        print("========FTC==========")
        print("k_attach_right_FTC = "+str(self.k_attach_right_FTC))
        print("k_attach_left_FTC = "+str(self.k_attach_left_FTC))
        print("k_detach_right_FTC = "+str(self.k_detach_right_FTC))
        print("k_detach_left_FTC = "+str(self.k_detach_left_FTC))
        print("center_FTC = "+str(self.center_FTC))
        print("spread_FTC = "+str(self.spread_FTC))
        
        print("========phenmenlogical rates==========")
        print("k_attach_left = "+str(self.k_attach_left_FTC*self.conc_FTC + self.k_attach_left_C * self.conc_C))
        print("k_attach_right = "+str(self.k_attach_right_FTC*self.conc_FTC + self.k_attach_right_C * self.conc_C))
        print("k_detach_left = "+str(self.k_detach_left_FTC*self.conc_ETC + self.k_detach_left_C))
        print("k_detach_right = "+str(self.k_detach_right_FTC*self.conc_ETC + self.k_detach_right_C))
        
        print("============concentration (chemical potential)===========")
        print("[FTC] = "+str(self.conc_FTC))
        print("[ETC] = "+str(self.conc_ETC))
        print("[C] = "+str(self.conc_C))
        sys.stdout.flush()
    def compute_k_left(self):
        x1 = np.arange(-self.well_width/2, self.well_width/2,0.02)[:,np.newaxis]
        x2 = np.arange(-self.well_width/2, self.well_width/2,0.02)[np.newaxis,:]
        U_bare1 = self.potential_bare_track(x1)
        U2 = self.potential_bare_track(x2)
        CAT = 2
        dx_CAT = x1 - CAT
        U_LJ = potential_LJ_(self.epA, self.epR, dx_CAT, self.sigma)
        U1 = U_bare1 + U_LJ
        dr = x1-(x2-self.shifted_distance)
        U_coup = self.potential_coupling(dr)
        U = U1 +U2+U_coup
        idx_x1 = np.where(U == np.min(U))[0][0] # indices of x
        self.k_close = np.exp(-self.beta * U_LJ[idx_x1])[0]
        
    def compute_prefactor(self):
        self.prefactor = np.exp(self.beta*(self.mu_C_std +self.E_C_track)) # the prefactor e^{beta(\mu_C^0 - E_{C-track})}
    def check_nan(self,x,string):
        if np.any(np.isnan(x)):
            print(string+" contains NaN values")
            sys.exit()
    def Fermi_function(self,x,center,spread,k_right=1, k_left=0):
        return (k_right-k_left)*(1/(1+np.exp((-x+center)/spread)))+k_left
        
    def Fit_fermi_function_for_expfactor(self):
        d = np.linspace(0.1,5,100)
        U = potential_LJ_(self.epA, self.epR, d,self.sigma) # repulsion between red particle and green ring
        initial_guess = [0, 1]  # Adjust these based on expected values
        params, _ = curve_fit(self.Fermi_function, d,np.exp(-self.beta*U), p0=initial_guess)
        # Extract fitted parameters
        self.center_C, self.spread_C = params
        print("center_C: "+str(self.center_C))
        print("spread_C: "+str(self.spread_C))
        
    def compute_k_detach_C(self):
        self.k_detach_right_C = self.k_attach_right_C/(self.k_far*self.prefactor)
        self.k_detach_left_C = self.k_attach_left_C/(self.k_close*self.prefactor)   
    
    
    def BIND_position(self):
        self.BIND1 = np.arange(0,self.tot_length,self.repeated_length,dtype =np.int32)
        self.BIND2 = np.arange(self.shifted_distance,self.shifted_distance+self.tot_length,self.repeated_length,dtype =np.int32)
        self.BIND1_ = np.arange(self.repeated_length/2,self.repeated_length+1,1,dtype = np.int32) # broader core
        self.BIND2_ = np.arange(self.repeated_length/2+self.shifted_distance,self.repeated_length+self.shifted_distance+1,1,dtype = np.int32) # broader core
        BIND1_broad = []
        for i in range(self.num_motifs):
            BIND1_broad.extend(self.BIND1_ +i*self.repeated_length)
        self.BIND1_broad = np.array(BIND1_broad)
        self.BIND1_broad[self.BIND1_broad >= self.tot_length] -= self.tot_length
        
        BIND2_broad = []
        for i in range(self.num_motifs):
            BIND2_broad.extend(self.BIND2_ +i*self.repeated_length)
        self.BIND2_broad = np.array(BIND2_broad)
        self.BIND2_broad[self.BIND2_broad >= self.tot_length] -= self.tot_length
        
    def CAT_position(self):
        CAT1 =np.arange(2,2+self.tot_length,self.repeated_length)
        self.CAT1_pos = np.tile(CAT1,(self.n,1))
        
        CAT2 = np.arange(2+self.shifted_distance,2+self.tot_length+self.shifted_distance,self.repeated_length)
        self.boundary_single(CAT2)
        self.CAT2_pos = np.tile(CAT2,(self.n,1))
        
    # if the particle is out of the bound move it back
    def boundary_single(self,x):
        x[x<0] += self.tot_length
        x[x >= self.tot_length] -= self.tot_length
        
    def boundary(self):
        self.boundary_single(self.x1)
        self.boundary_single(self.x2)
        
    # compute the minimal distance between the two particles
    def pbc(self,dx):
        dx[dx >= self.tot_length/2] -= self.tot_length
        dx[dx <= -self.tot_length/2] += self.tot_length
    # x2-x1
    def compute_dr(self):
        self.dr = self.x2-self.x1
        self.pbc(self.dr)
    # x - CAT
    def compute_dCAT(self,x,CAT_pos):
        # The CAT position for track 1 is: 2, 2 + 2*l, ..., 2+ (num_motif-1)*l, <-- apply boundary on CAT position, (fixed during the simulation)
        # The CAT position for track 2 is: 2 + shifted_distance,  <-- apply boundary on CAT position
        tiled_x = np.tile(x.reshape(-1,1),(1,self.num_motifs))
        dx_CAT = tiled_x-CAT_pos
        self.pbc(dx_CAT)
        return dx_CAT
    
    def compute_shifted_x_and_elementary_x(self):
        self.shifted_x1 = self.x1 +self.well_width/2
        self.boundary_single(self.shifted_x1)
        self.shifted_x2 = self.x2 +self.well_width/2-self.shifted_distance
        self.boundary_single(self.shifted_x2)
        self.elementary_shifted_x1 = self.shifted_x1%(self.repeated_length) -self.well_width/2
        self.elementary_shifted_x2 = self.shifted_x2%(self.repeated_length) -self.well_width/2
        
    # use fermi function to describe k_attach(r) and k_cleave(r)
    def k_attach_FTC(self,x):
        k = self.Fermi_function(x,self.center_FTC,self.spread_FTC,self.k_attach_right_FTC, self.k_attach_left_FTC)
        k[x < 2 + self.well_width/2] = 0
        return k
    
    def k_detach_FTC(self,x):
        return self.Fermi_function(x,self.center_FTC,self.spread_FTC,self.k_detach_right_FTC, self.k_detach_left_FTC)
    
    # def k_attach_C(self,x):
    #     return self.Fermi_function(x,self.center_C,self.spread_C,self.k_attach_right_C, self.k_attach_left_C)
    
    # def k_detach_C(self,x):
    #     return self.Fermi_function(x,self.center_C,self.spread_C,self.k_detach_right_C, self.k_detach_left_C)
    
    # this version does not rely on fermi function, but rely on the distance between ring and CAT sites
    def k_attach_C(self,x):
        U_LJ = potential_LJ_(self.epA, self.epR, x, self.sigma)
        return self.k_attach_right_C * np.exp(-self.beta*U_LJ*self.eta)
    
    def k_detach_C(self,x):
        U_LJ = potential_LJ_(self.epA, self.epR, x, self.sigma)
        return self.k_attach_C(x)/self.prefactor * np.exp(self.beta*U_LJ)
    
    # phenomenological rates:
    def k_attach_r(self,x):
        k_attach_rates = self.k_attach_FTC(x)*self.conc_FTC + self.k_attach_C(x) * self.conc_C
        return k_attach_rates

    def k_detach_r(self,x):
        k_detach_rates = self.k_detach_FTC(x)*self.conc_ETC + self.k_detach_C(x)
        return k_detach_rates
    
    def Monte_Carlo_single_particle(self, MC_rand, x, CAT_pos, choices, P,particle_ID,replica_idx):
        # The distance between the CAT and the current ring position. 
        distance_x_CAT = np.abs(self.compute_dCAT(x,CAT_pos))
        k_attach = self.k_attach_r(distance_x_CAT)
        P = self.dt * k_attach *self.MC_steps
        k_detach = self.k_detach_r(distance_x_CAT)
        # deal with states that are currently blocked 
        idx = np.argwhere(choices[:,:-1] == 1)
        P[tuple(idx.T)] = self.dt * k_detach[tuple(idx.T)] *self.MC_steps
        #attach_close_idx = np.where(distance_x_CAT[tuple(idx.T)] < 2.7)[0]
        # if len(attach_close_idx) > 0:
        #     print("distance: "+str(distance_x_CAT[tuple(idx.T)][attach_close_idx]))
        #     print("detach close rates: "+str(k_detach[tuple(idx.T)][attach_close_idx]))
        # check if the probability is too big
        sum_P = np.sum(P,axis = 1)
        if sum_P.max() > 0.1:
            print(" Too big! acuumulated probability for particle is "+str(particle_ID)+": "+str(sum_P.max()))
            max_particle = np.argmax(sum_P)
            print("Particle's potential: "+str(choices[max_particle]))
            print("Ring position: "+str(x[max_particle]))
            print("Probability: "+str(P[max_particle]))
            sys.exit()
        P = np.cumsum(P,axis = 1)
        flip_choice = np.diag(np.apply_along_axis(np.searchsorted, 1, P, MC_rand)).reshape(-1,1)#[[np.searchsorted(self.P1[i], MC_r1[i])] for i in range(len(MC_r1))]#
        choices[replica_idx,flip_choice] *= -1
        return choices, P
        
    def Monte_Carlo_step(self,cycle):
        replica_idx = np.arange(self.n).reshape(-1,1)
        if self.MC_x1_on == True:
            self.Monte_Carlo_single_particle(self.MC_rand1[cycle], self.x1, self.CAT1_pos, self.choices_1, self.P1,1,replica_idx)
        self.Monte_Carlo_single_particle(self.MC_rand2[cycle], self.x2, self.CAT2_pos, self.choices_2, self.P2,2,replica_idx)    
    
    # for testing purpoose, single track
    def potential_bare_track(self,x):
        return potential_bare_track_(x,self.well_width,self.barrier_height)
    
    def potential_LJ(self,dx_CAT,choices):
        # force from barriers
        U_all_barriers = potential_LJ_(self.epA, self.epR, dx_CAT, self.sigma)
        # deal with states that are currently blocked 
        blocked_unblocked = np.array(choices[:,:-1])
        blocked_unblocked[blocked_unblocked == -1] = 0
        U_LJ_sum = np.sum(U_all_barriers*blocked_unblocked,axis = 1)
        return U_LJ_sum
    
    def potential_coupling(self,dr = None):
        if dr is None:
            dr = self.dr
        if self.coupling_type == "spring":
            return potential_spring_(dr,self.k,self.x0,self.cross_distance)
        elif self.coupling_type == "FENE":
            return potential_FENE_(dr,self.k,self.x0,self.cross_distance)
        
    def potential_calculation(self):
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
        
    
    def coupling_force(self):
        if self.coupling_type == "spring":
            return force_spring_(self.dr,self.k,self.x0,self.cross_distance)
        elif self.coupling_type == "FENE":
            return force_FENE_(self.dr,self.k,self.x0,self.cross_distance)
    def LJ_force(self,dx_CAT,choices):
        # force from barriers
        f_LJ_all_barriers = force_LJ_(self.epA, self.epR, dx_CAT, self.sigma)
        # deal with states that are currently blocked 
        blocked_unblocked = np.array(choices[:,:-1])
        blocked_unblocked[blocked_unblocked == -1] = 0
        f_LJ_sum = np.sum(f_LJ_all_barriers*blocked_unblocked,axis = 1)
        
        return f_LJ_sum
    
    def force_calculation(self):
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
        
    def underdamped(self, rand1, rand2):
        #rand is a 3D array, length = (2, 2, self.n), first 2 is rand1 and rand2, second 2 is 2D dim of the space,
        self.p1=self.p1*self.expgamma+self.f1*self.dt/2+rand1[0]
        self.x1+=self.p1*self.dt/self.m
        # Check if any element is greater than 0
        if np.any(np.abs(self.p1*self.dt/self.m) > 1):
            print("increment too large for x1!")
            print(np.abs(self.p1*self.dt/self.m).max())

        self.p2=self.p2*self.expgamma+self.f2*self.dt/2+rand2[0]
        self.x2+=self.p2*self.dt/self.m
        self.boundary()
        if np.any(np.abs(self.p2*self.dt/self.m) > 1):
            print("increment too large for x2!")
            print(np.abs(self.p2*self.dt/self.m).max())

        self.force_calculation()
        self.p1=(self.p1+self.f1*self.dt/2)*self.expgamma+rand1[1]
        self.p2=(self.p2+self.f2*self.dt/2)*self.expgamma+rand2[1]     
    
    # I only care about the behavior of ring 1, if the ring is on the core positions, what the current potential is how many cycles have been performed.
    def compute_integer_x1_x2(self):
        self.x1_int = np.round(self.x1)
        self.x1_int[self.x1_int == self.tot_length] = 0
        self.x2_int = np.round(self.x2)
        self.x2_int[self.x2_int == self.tot_length] = 0
        
    def compute_new_x1_x2_core(self):
        self.compute_integer_x1_x2()
        x1_state = np.searchsorted(self.BIND1+3,self.x1_int) * self.repeated_length
        x2_state = np.searchsorted(self.BIND2+3,self.x2_int) * self.repeated_length
        x1_state[x1_state == self.tot_length] = 0
        x2_state[x2_state == self.tot_length] = 0
        return x1_state, x2_state
    
    def coarse_graining_states(self,step,option,transition_folder_name,transition_file_handles,idx_traj):
        if option == 0:
            # rings' position
            self.x1_state_old, self.x2_state_old  = self.compute_new_x1_x2_core()
            self.x1_state_new = np.array(self.x1_state_old)
            self.x2_state_new = np.array(self.x2_state_old)
            # ring 1's blocking status
            potential_type_1 = np.array(self.choices_1[:,:-1])
            self.core_old = np.concatenate((self.x1_state_old.reshape(-1,1),potential_type_1),axis = 1)
            self.core_new = np.array(self.core_old)
            self.wt_array = np.zeros(self.n)+self.dt
            flag_change = []
            self.recording_transitions(True,step,transition_folder_name,transition_file_handles,idx_traj,flag_change)
        elif option == 1:
            self.x1_state_new, self.x2_state_new  = self.compute_new_x1_x2_core()
            x1_not_in_core = np.where(~np.isin(self.x1_int, self.BIND1_broad))[0]
            x2_not_in_core = np.where(~np.isin(self.x2_int, self.BIND2_broad))[0]
            self.x1_state_new[x1_not_in_core] = self.x1_state_old[x1_not_in_core]
            self.x2_state_new[x2_not_in_core] = self.x2_state_old[x2_not_in_core]
            potential_type_1 = np.array(self.choices_1[:,:-1])
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
            
    # # only for counting hops
    # def checkpoint(self,init):
    #     # if option == 0, attribute x1 & x2 to a position with binary method
    #     if init == True: 
    #         self.x1_state_old = np.int32(self.x1/self.repeated_length)
    #         # shift x2 back!
    #         self.x2_state_old = np.int32((self.x2-self.shifted_distance)/self.repeated_length)
    #         self.x1_state_new = np.array(self.x1_state_old)
    #         self.x2_state_new = np.array(self.x2_state_old)
    #     else:
    #         self.x1_state_new = np.int32(self.x1/self.repeated_length)
    #         self.x2_state_new = np.int32((self.x2-self.shifted_distance)/self.repeated_length)
    #         dist_to_core_x1 = self.x1-self.x1_state_new * self.repeated_length
    #         self.pbc(dist_to_core_x1)
    #         dist_to_core_x1 = np.abs(dist_to_core_x1)
    #         index_inter_1 = np.where(dist_to_core_x1 > self.delta_gap)[0]
    #         self.x1_state_new[index_inter_1] = np.array(self.x1_state_old)[index_inter_1]
    #         dist_to_core_x2 = self.x2-self.shifted_distance-self.x2_state_new * self.repeated_length
    #         self.pbc(dist_to_core_x2)
    #         dist_to_core_x2 = np.abs(dist_to_core_x2)
    #         index_inter_2 = np.where(dist_to_core_x2 > self.delta_gap)[0]
    #         self.x2_state_new[index_inter_2] = np.array(self.x2_state_old)[index_inter_2]
    #         # check if the state is making right cycles
    #         self.right_cycles_x1[(self.x1_state_new - self.x1_state_old == 1) | (self.x1_state_new - self.x1_state_old == 1-self.num_motifs)]+= 1
    #         self.left_cycles_x1[(self.x1_state_new - self.x1_state_old == -1) | (self.x1_state_new - self.x1_state_old == -1 + self.num_motifs)]+= 1     
    #         self.right_cycles_x2[(self.x2_state_new - self.x2_state_old == 1) | (self.x2_state_new - self.x2_state_old == 1 - self.num_motifs)]+= 1
    #         self.left_cycles_x2[(self.x2_state_new - self.x2_state_old == -1) | (self.x2_state_new - self.x2_state_old == -1 + self.num_motifs)]+= 1
    #         self.x1_state_old = np.array(self.x1_state_new)
    #         self.x2_state_old = np.array(self.x2_state_new)
    # def core_states(self,step,option,transition_folder_name,transition_file_handles,idx_traj):
    #     if option == 0: 
    #         x1_core = np.searchsorted(self.core_pos,self.x1)
    #         x1_core[x1_core == self.num_motifs] = 0# I also care about whether position is in the middle
    #         x1_core = x1_core*np.int32(self.repeated_length)
    #         potential_type_1 = np.array(self.choices_1[:,:-1])
    #         self.core_old = np.concatenate((x1_core.reshape(-1,1),potential_type_1),axis = 1)
    #         self.core_new = np.array(self.core_old)
    #         self.wt_array = np.zeros(self.n)+self.dt
    #         flag_change = []
    #         self.recording_transitions(True,step,transition_folder_name,transition_file_handles,idx_traj,flag_change)
    #     elif option == 1:
    #         #### check if the state has updated#######
    #         #x1_core = np.int32(self.x1/(self.repeated_length))*np.int32(self.repeated_length)
    #         x1_core = np.searchsorted(self.core_pos,self.x1)
    #         x1_core[x1_core == self.num_motifs] = 0# I also care about whether position is in the middle
    #         x1_core = x1_core*np.int32(self.repeated_length)
    #         potential_type_1 = np.array(self.choices_1[:,:-1])
    #         self.core_new = np.concatenate((x1_core.reshape(-1,1),potential_type_1),axis = 1)
            
    #         dist_to_core = self.x1-x1_core
    #         self.pbc(dist_to_core)
    #         dist_to_core = np.abs(dist_to_core)
    #         ring_in_core = np.where(dist_to_core > self.delta_gap)[0]
    #         self.core_new[:,0][ring_in_core] = np.array(self.core_old[:,0])[ring_in_core] # only make ring positions core.
    #         flag_change = np.where(np.any(self.core_new -self.core_old != np.zeros(self.num_motifs+1),axis = 1))[0]
    #         self.wt_array += self.dt
    #         #### if so, doing the recordings##########
    #         self.recording_transitions(False,step,transition_folder_name,transition_file_handles,idx_traj,flag_change)
            
    #         self.wt_array[flag_change] = self.dt
    #         self.core_old = np.array(self.core_new)
    #     elif option ==2:
    #         for transition_file_handle in transition_file_handles:
    #             transition_file_handle.close()
            
    def recording_transitions(self,init,step,transition_folder_name,transition_file_handles,idx_traj,flag_change):
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
            
    def propagation_underdamped_diffusion_jump_motor(self, steps, idx_traj):
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
        c2 = 0
        ############debug############
        for i in range(steps):
            c1 = i%self.cycle
            if c1 == 0:
                #random numbers (with chuncks), otherwise the cluster cannot run it.
                self.rand1 = np.random.normal(0,self.stdx,(self.cycle,2,self.n)); # Particle 1
                self.rand2 = np.random.normal(0,self.stdx,(self.cycle,2,self.n)); # Particle 2
            self.underdamped(self.rand1[c1],self.rand2[c1])
            self.check_nan(self.x1,"self.x1")
            self.check_nan(self.x2,"self.x2")
            if (i+1)%self.MC_steps == 0:
                c2 = np.int32(i/self.MC_steps)%(self.cycle)
                if c2 == 0:
                    self.MC_rand1 = np.random.uniform(0,1,(self.cycle,self.n));
                    self.MC_rand2 = np.random.uniform(0,1,(self.cycle,self.n));
                self.Monte_Carlo_step(c2)
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
                                                self.right_cycles_x2,self.left_cycles_x2,np.transpose(self.choices_1[:,:-1]).flatten(),np.transpose(self.choices_2[:,:-1]).flatten())))
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
        folder_name = "simulation_data"
        if not os.path.exists(folder_name) and idx_traj ==1:
            os.makedirs(folder_name)
        folder_name = "transition_data"
        if not os.path.exists(folder_name) and idx_traj ==1:
            os.makedirs(folder_name)
        self.initialize_system()
        self.propagation_underdamped_diffusion_jump_motor(steps,idx_traj)

############ packed everything together##############
class DiffusionJumpMotor(diffusion_jump_motor):
    def __init__(self,
                 # equilibrium
                 MC_steps = 100,
                 # potential parameters
                 barrier_height=1.17, well_width=4, repeated_length=12, num_motifs=4,
                 # interaction parameters
                 epR=1e4, E_C_track=5.414,
                 # rate constant parameters
                 k_attach_right_FTC=2e-5, k_attach_left_FTC=0, k_detach_right_FTC=0, k_detach_left_FTC=0,
                 center_FTC=4.6, spread_FTC=0.5,
                 k_attach_right_C=3e-3, k_attach_left_C=1.2e-3,
                 # concentration parameters
                 conc_FTC=10, conc_ETC=0, conc_C=0,
                 # coupling parameters
                 shifted_distance=0, coupling_type="spring", coupling_strength=0, coupling_center=0,gamma = 6):
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
        self.E_C_track = E_C_track

        # Initialize rate constant parameters
        self.k_attach_right_FTC = k_attach_right_FTC
        self.k_attach_left_FTC = k_attach_left_FTC
        self.k_detach_right_FTC = k_detach_right_FTC
        self.k_detach_left_FTC = k_detach_left_FTC
        self.center_FTC = center_FTC
        self.spread_FTC = spread_FTC
        self.k_attach_right_C = k_attach_right_C
        self.k_attach_left_C = k_attach_left_C

        # Initialize concentration parameters
        self.conc_FTC = conc_FTC
        self.conc_ETC = conc_ETC
        self.conc_C = conc_C

        # Initialize coupling parameters
        self.shifted_distance = shifted_distance
        self.coupling_type = coupling_type
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
        idx_list = np.array([i for i in range(0, parallel_jobs, 1)])
        Parallel(n_jobs=int(parallel_jobs))(delayed(self.run_simulation)(trialID+1,steps) for trialID in idx_list)
        
if __name__ == "__main__":
    motor = diffusion_jump_motor()
    motor.k = 0
    motor.x0 = 0
    motor.shifted_distance = 0
    motor.coupling_type = "spring"
    motor.k_attach_right_FTC = 0.002
    motor.parallel_propagation_underdamped_diffusion_jump_motor(idx_traj=1, steps=100000)

