# Only jot down the neighboring C occupancies of the rings 
# for simulation since 11/23/2023
import numpy as np
import sys
# quantities for all trials
PATH = sys.argv[1]

num_trial = 100
trial = np.arange(2,num_trial+1,1,dtype = np.int32)

num_motif = 4
length_motif = 12
num_bead = num_motif*length_motif
burning_steps = 4000
#get the neighboring blocking status of Ring 1 and Ring 2. 
# Ring 2 might be offset from Ring 1, so I need to shift Ring 2 to Ring 1 positioning to find the corresponding
# 3 neighboring CAT groups 
def neighboring_blocking_group(data,num_motif,length_motif):
    ring_pos1 = data[:, 3:4]
    C1 = data[:,4:8]
    
    # Find the Binding indices of the two rings for each states
    BIND_idx1 = np.int32(data[:,3]/length_motif)

    # So the corresponding CAT indices are
    CAT_indices1 = np.transpose(np.array([BIND_idx1 -1, BIND_idx1, BIND_idx1+1]))
    # Add PBC to shift CAT indices so that they are between 0 and num_motif-1
    idx = np.argwhere(CAT_indices1 < 0)
    CAT_indices1[tuple(idx.T)] += num_motif
    idx = np.argwhere(CAT_indices1 > num_motif-1)
    CAT_indices1[tuple(idx.T)] -= num_motif

    # Extract the neighboring C
    row_indices = np.arange(len(CAT_indices1)).reshape(-1, 1)
    C1_neighbor = C1[row_indices,CAT_indices1]
    cutoff_states = np.concatenate([ring_pos1, C1_neighbor], axis=1)
    return cutoff_states


def read_a_traj(trial_num):
    print("trial: "+str(trial_num))
    path = PATH +"/transition_data/" +"%04d.txt"%(trial_num)
    data = np.loadtxt(open(path))
    print("Finished Loading the data...")
    states_cutoff = neighboring_blocking_group(data,num_motif,length_motif)
    return states_cutoff

unique_states = read_a_traj(1)

for j in trial:
    states = read_a_traj(j)
    unique_states = np.concatenate((unique_states,states))
unique_states,counts = np.unique(unique_states, axis=0,return_counts = True)

combined_array = np.hstack((unique_states, np.array([counts]).reshape(-1,1)))
np.savetxt(PATH+'/'+"unique_states_CG.txt",combined_array, fmt='%d')

##########################define good states, pruning the bad states############################################
good_centers = unique_states[counts > 10]
np.savetxt(PATH+'/'+"good_centers_CG.txt",good_centers, fmt='%d')
