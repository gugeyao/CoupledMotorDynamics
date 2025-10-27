# do linear algebra to compute the steady-state distribution and the current from Transition rate matrix (TRM)
import numpy as np
from scipy.sparse import csr_matrix 
import sys
path =  sys.argv[1]#'/home/ggu7596/project/optimal_control/motorsim_double_track/data/week13year2024/Double_track_spring_simulation_bug_fixed/long_track/phase_0/0_10'
num_bead = np.int32(sys.argv[2])
num_sets = 4
states_file = 'good_centers_CG.txt'
states = np.loadtxt(open(path +'/'+states_file))
keys = ["_CG"]
current_array = np.zeros(len(keys))
for u,key in enumerate(keys):
    rates_file = 'TRM'+key+'.txt'
    rate_matrix_non_zero = np.loadtxt(open(path +'/'+rates_file))
    Population_file = 'Population'+key+'.txt'
    #Population_err_file = 'Population_err.txt'
    population = np.loadtxt(open(path +'/'+ Population_file))
    rows=rate_matrix_non_zero[:,0]
    cols = rate_matrix_non_zero[:,1]
    vals = rate_matrix_non_zero[:,2]
    # creating sparse matrix 
    rate_matrix = csr_matrix((vals, (rows, cols)),  
                            shape = (len(states), len(states))).toarray() 
    A = np.transpose(rate_matrix)
    print("Det: ")
    print(np.linalg.det(A))
    B = np.ones((len(rate_matrix)))*1e-18
    C= np.linalg.solve(A,B)
    pi = C/np.sum(C)
    
    print("Finished computing the steady-state population...")
    #################################compute the edge-current ########################
    #new_rate_matrix = np.array(rate_matrix)
    #np.fill_diagonal(new_rate_matrix, 0)
    #flux_matrix = new_rate_matrix*np.outer(pi,np.ones(len(pi)))
    #current_matrix = np.maximum(flux_matrix - flux_matrix.T, 0)
    
    # Compute the current from the Markov model
    indicator_matrix = np.zeros_like(rate_matrix) # indicate if the current is generated or not
    ring_pos1 = states[:,0]
    for k in range(len(rows)):
        i = np.int32(rows[k])
        j = np.int32(cols[k])
        ring_pos_i = ring_pos1[i]
        ring_pos_j = ring_pos1[j]
        delta = ring_pos_j - ring_pos_i
        if delta >num_bead*num_sets/2:
            delta -= num_bead*num_sets
        if delta <-num_bead*num_sets/2:
            delta += num_bead*num_sets
        indicator_matrix[i][j] = delta*rate_matrix[i][j]*pi[i]
    edge_current = np.sum(indicator_matrix)
    current_array[u] = edge_current
    print("The current from MSM is: "+str(edge_current))
    #current_file = "current.txt"
    #current_data = np.loadtxt(open(path+"/"+current_file))
    #current = current_data[0]
    #current_err = current_data[1]
    #print("The current from the direct simulation is: "+str(current)+"+/-"+str(current_err))

    #################### save files ####################
    np.savetxt(path+"/current_MSM"+key+".txt",[edge_current])
    np.savetxt(path +"/computed_population"+key+".txt",pi)
    #np.savetxt(path+"/current_matrix"+key+".txt", current_matrix)
print("####################################")
if len(keys) > 1:
    current_ave = np.mean(current_array)
    current_err = np.std(current_array)/np.sqrt(len(keys)-1)
    print("Averaged current from MSM is: "+str(current_ave)+", and the std err is: "+str(current_err))
