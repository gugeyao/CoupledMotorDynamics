# This version of computing the transition rate matrix greately saves the computational space
import numpy as np
from scipy.sparse import csr_array
import sys
num_trial = 100 # number of trials
groups = 1
trials = np.arange(0,num_trial,1)
#load the states of each transitions to compute the number of states
PATH = sys.argv[1]#'/home/ggu7596/project/optimal_control/motorsim_double_track/data/week13year2024/Double_track_spring_simulation_bug_fixed/long_track/phase_0/0.01_10'
good_centers = 'good_centers_CG.txt'
good_center_data = np.loadtxt(open(PATH +'/'+good_centers))
num_states = len(good_center_data)
zero_array = np.array([0])
wt_array = np.zeros((groups, num_states)) # waiting times
count_matrix_array = [] # the groups of matrices of counting transitions
group_idx = -1
for i in range(groups):
    count_matrix = csr_array((zero_array, (zero_array, zero_array)), shape=(num_states, num_states),dtype = np.int32)
    count_matrix_array.append(count_matrix)

# read in the transition file for all trials. The format in the transition files is [label, waiting time]
#################################compute the mean TRM start#########################################################
for j in trials:
    if j%np.int32(num_trial/groups) == 0:
        group_idx+=1
    print("trial: "+str(j+1))
    #load data
    data = np.loadtxt(open(PATH +"/transition_data/" +"%04d_labeled.txt"%(j+1)))
    states_seq = np.array(data[:,0],dtype = np.int32) # states in the sequence and plus waiting time
    wt = data[:,1]
    ##################### compute the number of transitions############################
    state_i = states_seq[:-1]
    state_j = states_seq[1:]
    # only record transitions that are between different states
    ones = np.ones(len(state_i))
    count_matrix_array[group_idx] += csr_array((ones, (state_i, state_j)), shape=(num_states, num_states),dtype = np.int32)
    ##################### compute the total waiting time############################
    wt_array[group_idx,:len(np.bincount(state_i,  weights=wt[:-1]))]+= np.bincount(state_i,  weights=wt[:-1])
##################### compute the rate matrix for each group##################
for group in range(groups):
    zero_idx = np.where(wt_array[group] ==0)[0]
    print("zero idx: "+str(zero_idx))
    print("those states are: "+str(good_center_data[zero_idx]))
    #print(wt_array[group])
    TRM = count_matrix_array[group]/wt_array[group][:, np.newaxis]
    # set diagnal elements to zeros
    TRM_dense = TRM.toarray()
    np.fill_diagonal(TRM_dense, 0)
    row_sums = np.sum(TRM_dense, axis=1)
    TRM_dense -= np.diag(row_sums)
    print(TRM_dense.min())
    #################################compute the mean TRM end#########################################################
    #################################compute the mean Population ############################################
    population = wt_array[group]/np.sum(wt_array[group])
    #########################store data##########################
    non_zero_elements = TRM_dense[TRM_dense != 0]
    non_zero_rows, non_zero_cols = np.where(TRM_dense != 0)
    TRM_nonzero = np.transpose([non_zero_rows, non_zero_cols, non_zero_elements])
    np.savetxt(PATH+"/TRM_CG.txt",TRM_nonzero, fmt=('%d','%d','%1.18f'))
    np.savetxt(PATH+"/Population_CG.txt",population, fmt='%1.18f')
