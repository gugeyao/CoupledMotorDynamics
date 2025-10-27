# for a waiting time, want to put the ring position to the exact same bead
import numpy as np

from deeptime.clustering import ClusterModel 
import warnings
from numpy import linalg as LA

warnings.filterwarnings("ignore")
import time
import sys
PATH = sys.argv[2]#'/home/ggu7596/project/optimal_control/motorsim_double_track/data/week13year2024/Double_track_spring_simulation_bug_fixed/long_track/phase_0/0.01_10'
num_motif = 4
length_motif = 12
num_bead = num_motif*length_motif
burning_steps = 4000
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

trial_num = np.int32(sys.argv[1])
path = PATH +"/transition_data/" +"%04d.txt"%(trial_num)
print(path)
data = np.loadtxt(open(path))
waiting_time = data[:,1]

#########################import centers
good_centers = np.loadtxt(open(PATH+"/good_centers_CG.txt"))

states_cutoff = neighboring_blocking_group(data,num_motif,length_motif)
#print(states_cutoff)

#label = pyemma.coordinates.assign_to_centers(states_cutoff,good_centers)[0]
#print(good_centers[label])
model = ClusterModel(cluster_centers=good_centers)  # Use good_centers as fixed
label = model.transform(states_cutoff)  # Returns cluster indices
transition_ = np.transpose(np.array([label,waiting_time]))
np.savetxt(PATH +"/transition_data/" +"%04d_labeled.txt"%(trial_num),transition_)

#np.savetxt(path +'/'+'transition_with_label_CG.txt',transition_, fmt=('%d','%1.18f'))
