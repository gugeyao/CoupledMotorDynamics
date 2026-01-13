import numpy as np
from functools import reduce
# I use this class to post process data and plot

class nonequilibrium_analysis(object):    
    def __init__(self):
        self.num_trials = 100
        self.repeated_length = 12
        self.num_motifs = 4
    def temperature_check(self,path):
        a_folder = path + "/simulation_data"
        T_array = np.zeros(self.num_trials)
        for k in range(self.num_trials):
            file = "%04d.txt" % (k + 1)
            data = np.loadtxt(open(a_folder + "/" + file))
            p1 = data[:,3]
            m = 12
            p1_mean = np.mean(p1)
            T1 = np.mean(p1**2)/m
            T_array[k] = T1
        T_mean = np.mean(T_array)
        T_err = np.std(T_array)/np.sqrt(self.num_trials)
        return T_mean, T_err
    def calculate_current(self, path, length = None, num_trial = None):
        if length is None:
            length = self.repeated_length
        if num_trial is None:
            num_trial = self.num_trials
        # Path to the folder containing the data files
        a_folder = path + "/transition_data"
        # Initialize the array to store currents
        current_array = np.full(num_trial, np.nan)  # Start with all values as NaN
        # Loop through each trial
        for k in range(num_trial):
            # Construct the filename and print the current trial
            file = "%04d.txt" % (k + 1)
            try:
                # Load the data from the file
                data = np.loadtxt(open(a_folder + "/" + file))
                # Skip empty files (loadtxt returns an empty array if no data is present)
                if data.size == 0:
                    raise ValueError(f"It contains no data.")

                # Truncate the data to exclude the last row
                # Ensure the data has sufficient columns to avoid indexing errors
                if data.ndim < 2 or data.shape[1] < 10:
                    raise ValueError(f"Not have enough columns.")

                # Extract relevant columns
                t = data[:, 0]
                right_cycle = data[:, 8]
                left_cycle = data[:, 9]

                # Calculate the current using the given formula
                current = ((right_cycle[-1]) - (left_cycle[-1])) / (t[-1])

                # Store the current value adjusted by length in the current array
                current_array[k] = current * length

            except (ValueError, IndexError) as e:
                # Skip files that raise a ValueError or IndexError
                print("file path = "+str(a_folder + "/" + file))
                print(f"Skipping file {file} due to error: {e}")

        # Calculate the mean and standard error, ignoring NaN values
        current_mean = np.nanmean(current_array)  # Average over non-NaN values
        current_err = np.nanstd(current_array) / np.sqrt(np.count_nonzero(~np.isnan(current_array)))  # Standard error over non-NaN values
        return current_mean, current_err
    
    def hopping_rates(self,path, num_trial = None):
        if num_trial is None:
            num_trial = self.num_trials
        # Path to the folder containing the data files
        a_folder = path + "/transition_data"

        # Initialize arrays to store hopping rates
        r_r_array = np.full(num_trial, np.nan)  # Start with NaN values to handle missing data
        r_l_array = np.full(num_trial, np.nan)

        # Loop through each trial
        for k in range(num_trial):
            # Construct the filename and print the current trial
            file = "%04d.txt" % (k + 1)
            try:
                # Load the data from the file
                data = np.loadtxt(open(a_folder + "/" + file))

                # Skip empty files (loadtxt returns an empty array if no data is present)
                if data.size == 0:
                    print(f"It contains no data.")
                    continue

                # Extract the last row of the data
                data = data[-1]

                # Ensure the data has at least enough elements to avoid indexing errors
                if data.shape[0] < 3:
                    raise ValueError(f"Not have enough columns.")

                # Extract right and left hops and time
                t = data[0]
                right_hops = data[-2]
                left_hops = data[-1]

                # Avoid division by zero in time
                if t == 0:
                    print(f"Skipping file {file} due to zero time value.")
                    continue

                # Calculate hopping rates
                r_r_array[k] = right_hops / t
                r_l_array[k] = left_hops / t

            except (ValueError, IndexError) as e:
                # Skip files that raise a ValueError or IndexError
                print(f"Skipping file {file} due to error: {e}")

        # Calculate the mean and standard error, ignoring NaN values
        r_r_mean = np.nanmean(r_r_array)  # Average over non-NaN values
        r_r_err = np.nanstd(r_r_array) / np.sqrt(np.count_nonzero(~np.isnan(r_r_array)))  # Standard error over non-NaN values

        r_l_mean = np.nanmean(r_l_array)  # Average over non-NaN values
        r_l_err = np.nanstd(r_l_array) / np.sqrt(np.count_nonzero(~np.isnan(r_l_array)))  # Standard error over non-NaN values

        return r_r_mean, r_r_err, r_l_mean, r_l_err
    def population_sites_blocked(self,path,shiftedDistance):
        a_folder = path +"/simulation_data"
        left_occ_ratio1 = np.zeros(self.num_trials)
        right_occ_ratio1 = np.zeros(self.num_trials)
        left_occ_ratio2 = np.zeros(self.num_trials)
        right_occ_ratio2 = np.zeros(self.num_trials)
        for k in range(self.num_trials):
            file = "%04d.txt"%(k+1)
            #print(k)
            data = np.loadtxt(open(a_folder+"/"+file))
            x1 = data[:,1]
            x2 = data[:,2]
            occ1 = data[:,-8:-4]
            occ2 = data[:,-4:]
            x1 = x1 -2
            x1[x1 < 0] += self.repeated_length*4
            x1_BIND = np.int32(x1/self.repeated_length)
            
            x2 = x2 -2-shiftedDistance
            x2[x2 < 0] += self.repeated_length*4
            x2_BIND = np.int32(x2/self.repeated_length)
            deltax = x2_BIND-x1_BIND
            deltax[deltax >= 2] -=4
            deltax[deltax <= -2] +=4
            x2Behindx1 = np.where(deltax < 0)[0]
            
            x1 = x1[x2Behindx1]
            x2 = x2[x2Behindx1]
            occ1 = occ1[x2Behindx1]
            occ2 = occ2[x2Behindx1]
            x1_BIND = x1_BIND[x2Behindx1]
            x2_BIND = x2_BIND[x2Behindx1]
            x1_BIND_right = np.int32(x1/self.repeated_length)+1
            x1_BIND_right[x1_BIND_right>=4] = 0
            x1_indices = np.arange(0,len(x1),1).reshape(-1,1)
            occ_left = occ1[x1_indices,x1_BIND.reshape(-1,1)]
            occ_right = occ1[x1_indices,x1_BIND_right.reshape(-1,1)]
            left_occupied = np.where(occ_left ==1)[0]
            right_occupied = np.where(occ_right ==1)[0]
            left_occ_ratio1[k] = len(left_occupied)/len(x1)
            right_occ_ratio1[k] = len(right_occupied)/len(x1)
            
            x2_BIND_right = np.int32(x2/self.repeated_length)+1
            x2_BIND_right[x2_BIND_right>=4] = 0
            x2_indices = np.arange(0,len(x2),1).reshape(-1,1)
            occ_left = occ2[x2_indices,x2_BIND.reshape(-1,1)]
            occ_right = occ2[x2_indices,x2_BIND_right.reshape(-1,1)]
            left_occupied = np.where(occ_left ==1)[0]
            right_occupied = np.where(occ_right ==1)[0]
            left_occ_ratio2[k] = len(left_occupied)/len(x2)
            right_occ_ratio2[k] = len(right_occupied)/len(x2)
            
        mean_left_occ_ratio1 = np.mean(left_occ_ratio1)
        err_left_occ_ratio1 = np.std(left_occ_ratio1)/np.sqrt(self.num_trials)
        mean_right_occ_ratio1 = np.mean(right_occ_ratio1)
        err_right_occ_ratio1 = np.std(right_occ_ratio1)/np.sqrt(self.num_trials)
        mean_left_occ_ratio2 = np.mean(left_occ_ratio2)
        err_left_occ_ratio2 = np.std(left_occ_ratio2)/np.sqrt(self.num_trials)
        mean_right_occ_ratio2 = np.mean(right_occ_ratio2)
        err_right_occ_ratio2 = np.std(right_occ_ratio2)/np.sqrt(self.num_trials)
        return mean_right_occ_ratio1,err_right_occ_ratio1,mean_left_occ_ratio1,err_left_occ_ratio1,\
                mean_right_occ_ratio2,err_right_occ_ratio2,mean_left_occ_ratio2,err_left_occ_ratio2
class equilibrium_analysis(nonequilibrium_analysis):
    def __init__(self):
        super().__init__()
    
    def histogram_dist(self,path):
        self.total_length = self.repeated_length *self.num_motifs
        a_folder = path +"/simulation_data"
        #bins = np.linspace(-self.repeated_length*self.num_motifs/2,self.repeated_length*self.num_motifs/2, num_bins)
        #P = np.zeros((self.num_trials,num_bins-1))
        bins = np.linspace(-self.total_length/2-0.5,self.total_length/2+0.5, self.total_length+2)
        P = np.zeros((self.num_trials,self.total_length+1))
        for k in range(self.num_trials):
            ############load data############
            file = "%04d.txt"%(k+1)
            #print(k)
            data = np.loadtxt(open(a_folder+"/"+file))
            x1 = data[:,1]
            x2 = data[:,2]
            delta_x = x2-x1
            delta_x[delta_x > self.repeated_length*self.num_motifs/2] -= self.repeated_length*self.num_motifs
            delta_x[delta_x < -self.repeated_length*self.num_motifs/2] += self.repeated_length*self.num_motifs
            #print(len(P[k]))
            #print(len(np.histogram(delta_x, bins=bins,density = True)))
            P[k],edges = np.histogram(delta_x, bins=bins,density = True)
        center = (edges[1:] +edges[:-1])/2
        P_mean = np.mean(P, axis = 0)
        P_err = np.std(P, axis = 0)/np.sqrt(self.num_trials)
        print(len(P_mean))
        return P_mean,P_err,center
    def histogram_dist_multibatches(self,path):
        self.total_length = self.repeated_length *self.num_motifs
        a_folder = path +"/simulation_data"
        #bins = np.linspace(-self.repeated_length*self.num_motifs/2,self.repeated_length*self.num_motifs/2, num_bins)
        #P = np.zeros((self.num_trials,num_bins-1))
        bins = np.linspace(-self.total_length/2-0.5,self.total_length/2+0.5, self.total_length+2)
        P = np.zeros((self.num_trials,self.total_length+1))
        groups = 10
        number_in_group = np.int32(self.num_trials/groups)
        for k in range(self.num_trials):
            ############load data############
            file = "%04d.txt"%(k+1)
            #print(k)
            data = np.loadtxt(open(a_folder+"/"+file))
            x1 = data[:,1]
            x2 = data[:,2]
            delta_x = x2-x1
            delta_x[delta_x > self.repeated_length*self.num_motifs/2] -= self.repeated_length*self.num_motifs
            delta_x[delta_x < -self.repeated_length*self.num_motifs/2] += self.repeated_length*self.num_motifs
            #print(len(P[k]))
            #print(len(np.histogram(delta_x, bins=bins,density = True)))
            P[k],edges = np.histogram(delta_x, bins=bins,density = True)
        center = (edges[1:] +edges[:-1])/2
        P = P[:groups*number_in_group].reshape(groups, number_in_group, -1)
        # Compute the mean and standard deviation for each group across all columns
        P_mean = P.mean(axis=1)  # Mean across rows in each group
        P_err = P.std(axis=1)/np.sqrt(number_in_group)    # Std across rows in each group
        print(len(P_mean))
        return P_mean,P_err,center
    def histogram_dist_x1(self,path,num_bins):
        self.total_length = self.repeated_length *self.num_motifs
        a_folder = path +"/simulation_data"
        bins = np.linspace(0,self.repeated_length*self.num_motifs, num_bins)
        P = np.zeros((self.num_trials,num_bins-1))
        # bins = np.linspace(-self.total_length/2-0.5,self.total_length/2+0.5, self.total_length+2)
        # P = np.zeros((self.num_trials,self.total_length+1))
        for k in range(self.num_trials):
            ############load data############
            file = "%04d.txt"%(k+1)
            #print(k)
            data = np.loadtxt(open(a_folder+"/"+file))
            x1 = data[:,1]
            #x2 = data[:,2]
            #delta_x = x2-x1
            #delta_x[delta_x > self.repeated_length*self.num_motifs/2] -= self.repeated_length*self.num_motifs
            #delta_x[delta_x < -self.repeated_length*self.num_motifs/2] += self.repeated_length*self.num_motifs
            #print(len(P[k]))
            #print(len(np.histogram(delta_x, bins=bins,density = True)))
            P[k],edges = np.histogram(x1, bins=bins,density = True)
        center = (edges[1:] +edges[:-1])/2
        P_mean = np.mean(P, axis = 0)
        P_err = np.std(P, axis = 0)/np.sqrt(self.num_trials)
        print(len(P_mean))
        return P_mean,P_err,center

class rates_analysis(nonequilibrium_analysis):
    def __init__(self):
        super().__init__()
    def calculate_transition_rates(self,path):
        a_folder = path +"/transition_data"
        k_ring_forward_array = []
        k_ring_back_array = []
        k_fuel_attach_array = []
        for k in range(self.num_trials):
            ############load data############
            file = "%04d.txt"%(k+1)
            #print(k)
            data = np.loadtxt(open(a_folder+"/"+file))
            if len(data.shape) == 2:
                wt = data[:,1]
                states = data[:,3:6] # I don't care what happened on the last and the second last catalytic site
                ############shuttling forward... ################
                state_i = np.array([0,-1,1])
                idx_i = np.where(np.all(states == state_i,axis = 1))[0]
                time_in_i = np.sum(wt[idx_i])
                state_j = np.array([self.repeated_length,-1,1])
                #state_middle_blocked = np.array([0,1,1])
                state_middle_blocked_ring_shuttled = np.array([self.repeated_length,1,1])
                
                idx_j = np.where(np.all(states == state_j,axis = 1))[0]
                #idx_middle_blocked = np.where(np.all(states == state_middle_blocked,axis = 1))[0]
                #idx_middle_blocked_ring_shuttled= np.where(np.all(states == state_middle_blocked_ring_shuttled,axis = 1))[0]
                num_ring_forward = len(np.array(list(set(idx_i) & set(idx_j-1))))
                #num_middle_blocked= len(np.array(list(set(idx_i) & set(idx_middle_blocked-1) & set(idx_middle_blocked_ring_shuttled-2))))
                if time_in_i !=0:
                    #k_ring_forward = (num_ring_forward+num_middle_blocked)/time_in_i
                    k_ring_forward = num_ring_forward/time_in_i
                    k_ring_forward_array.append(k_ring_forward)
                ############after shuttling forward... ################
                state_i = np.array([self.repeated_length,-1,1])
                state_ring_back = np.array([0,-1,1])
                state_fuel_attach = np.array([self.repeated_length,1,1])
                
                idx_i = np.where(np.all(states == state_i,axis = 1))[0]
                
                time_in_i = np.sum(wt[idx_i])
                
                idx_ring_back = np.where(np.all(states == state_ring_back,axis = 1))[0]
                idx_fuel_attach = np.where(np.all(states == state_fuel_attach,axis = 1))[0]
                
                num_ring_back = len(np.array(list(set(idx_i) & set(idx_ring_back-1))))
                num_fuel_attach  = len(np.array(list(set(idx_i) & set(idx_fuel_attach-1))))
                if time_in_i !=0:
                    k_ring_back = num_ring_back/time_in_i
                    k_fuel_attach = num_fuel_attach/time_in_i
                    k_ring_back_array.append(k_ring_back)
                    k_fuel_attach_array.append(k_fuel_attach)
        k_ring_forward_mean = np.mean(k_ring_forward_array)
        k_ring_forward_err = np.std(k_ring_forward_array)/np.sqrt(len(k_ring_forward_array))   
        k_ring_back_mean = np.mean(k_ring_back_array)
        k_ring_back_err = np.std(k_ring_back_array)/np.sqrt(len(k_ring_back_array))  
        k_fuel_attach_mean = np.mean(k_fuel_attach_array)
        k_fuel_attach_err = np.std(k_fuel_attach_array)/np.sqrt(len(k_fuel_attach_array))  
        
        return k_ring_forward_mean, k_ring_forward_err, k_ring_back_mean,k_ring_back_err,k_fuel_attach_mean,k_fuel_attach_err
    def calculate_transition_rates_v2(self,path):
        a_folder = path +"/transition_data"
        k_ring_array = []
        for k in range(self.num_trials):
            file = "%04d.txt"%(k+1)
            data = np.loadtxt(open(a_folder+"/"+file))
            if len(data.shape) == 2:
                wt = data[:,1]
                replica_idx = np.arange(len(wt)).reshape(-1,1)
                Ring1 = data[:,3]
                Ring1_idx = np.int32(Ring1.reshape(-1,1)/self.repeated_length)
                Ring1_idx[Ring1_idx == self.num_motifs] = 0
                Ring1_change = Ring1_idx[1:]-Ring1_idx[:-1]
                Ring1_increase_idx = np.where((Ring1_change == 1) | (Ring1_change == 1-self.num_motifs))[0]
                C1 = data[:,4:8]
                C1[C1 < 0] = 0
                C1_close = C1[replica_idx,Ring1_idx]
                num_increase = len(Ring1_increase_idx)
                in_0 = np.where(np.all(C1_close == 0,axis = 1))[0]  # not blocked close
                time_in_0 = np.sum(wt[in_0])
                if time_in_0!= 0:
                    k_ring_array.append(num_increase/time_in_0)
        k_ring_array = np.array(k_ring_array)
        k_ring_mean = np.mean(k_ring_array)
        k_ring_err = np.std(k_ring_array)/np.sqrt(len(k_ring_array))
        return k_ring_mean,k_ring_err
    def k_attach_r(self,path):
        a_folder = path +"/transition_data"
        k_CAT_Ring_array = []
        unique_distance = np.array([2,10,14,22])
        bins_ = np.array([0,3,11,15,23], dtype=float)
        L = self.num_motifs * self.repeated_length
        halfL = L / 2.0
        CAT_pos_all = 2 + np.arange(self.num_motifs, dtype=np.int32) * self.repeated_length
        nbins = len(bins_) - 1
        for k in range(self.num_trials):
            file = "%04d.txt"%(k+1)
            data = np.loadtxt(open(a_folder+"/"+file))
            if len(data.shape) == 2 and data.size > 0:
                # compute waiting times from time differences (align to rows)
                t = data[:,0].astype(float)
                wt = np.empty_like(t)
                wt[0] = t[0]
                if wt.shape[0] > 1:
                    wt[1:] = t[1:] - t[:-1]
                Ring1 = data[:,3].astype(float)
                C_all = data[:, 4:4+self.num_motifs].copy()
                C_all[C_all < 0] = 0
                C_all = C_all.astype(np.int8)

                # Distances from coarse-grained ring state to each CAT, with PBC wrapping
                dist = np.abs(Ring1[:, None] - CAT_pos_all[None, :])
                dist = np.where(dist > halfL, L - dist, dist)
                # Bin indices per row and motif (match numpy.histogram behavior)
                bin_idx = np.digitize(dist, bins_, right=False) - 1  # 0..nbins-1 ideally
                bin_idx = np.clip(bin_idx, 0, nbins - 1)

                # Accumulate waiting time in bins when site is unblocked
                wt_bins = np.zeros((self.num_motifs, nbins), dtype=float)
                unblocked = (C_all == 0)
                for j in range(self.num_motifs):
                    mask = unblocked[:, j]
                    if np.any(mask):
                        b = bin_idx[mask, j]
                        w = wt[mask]
                        np.add.at(wt_bins[j], b, w)

                # Count 0->1 attachment events, attributing to pre-event ring state
                num_events = np.zeros_like(wt_bins)
                if C_all.shape[0] > 1:
                    attach_mask = (C_all[:-1] == 0) & (C_all[1:] == 1)
                    Ring1_prev = Ring1[:-1]
                    dist_prev = np.abs(Ring1_prev[:, None] - CAT_pos_all[None, :])
                    dist_prev = np.where(dist_prev > halfL, L - dist_prev, dist_prev)
                    bin_prev = np.digitize(dist_prev, bins_, right=False) - 1
                    bin_prev = np.clip(bin_prev, 0, nbins - 1)
                    for j in range(self.num_motifs):
                        idx = attach_mask[:, j]
                        if np.any(idx):
                            b = bin_prev[idx, j]
                            np.add.at(num_events[j], b, 1)

                # Rates per motif/bin
                rates = num_events / (wt_bins + 1e-10)
                # Guard bins with negligible time
                rates[wt_bins <= 1e-5] = np.nan

                # Collect per-motif rows to match previous API
                for j in range(self.num_motifs):
                    k_CAT_Ring_array.append(rates[j])
        k_CAT_Ring_array = np.array(k_CAT_Ring_array)
        return unique_distance,k_CAT_Ring_array
    def k_detach_r(self,path):
        a_folder = path +"/transition_data"
        k_CAT_Ring_array = []
        unique_distance = np.array([2,10,14,22])
        bins_ = np.array([0,3,11,15,23], dtype=float)
        L = self.num_motifs * self.repeated_length
        halfL = L / 2.0
        CAT_pos_all = 2 + np.arange(self.num_motifs, dtype=np.int32) * self.repeated_length
        nbins = len(bins_) - 1
        for k in range(self.num_trials):
            file = "%04d.txt"%(k+1)
            data = np.loadtxt(open(a_folder+"/"+file))
            if len(data.shape) == 2 and data.size > 0:
                # compute waiting times from time differences (align to rows)
                t = data[:,0].astype(float)
                wt = np.empty_like(t)
                wt[0] = t[0]
                if wt.shape[0] > 1:
                    wt[1:] = t[1:] - t[:-1]
                Ring1 = data[:,3].astype(float)
                C_all = data[:, 4:4+self.num_motifs].copy()
                C_all[C_all < 0] = 0
                C_all = C_all.astype(np.int8)

                # Distances with PBC wrapping
                dist = np.abs(Ring1[:, None] - CAT_pos_all[None, :])
                dist = np.where(dist > halfL, L - dist, dist)
                # Bin indices per row and motif (match numpy.histogram behavior)
                bin_idx = np.digitize(dist, bins_, right=False) - 1
                bin_idx = np.clip(bin_idx, 0, nbins - 1)

                # Accumulate waiting time in bins when site is blocked
                wt_bins = np.zeros((self.num_motifs, nbins), dtype=float)
                blocked = (C_all == 1)
                for j in range(self.num_motifs):
                    mask = blocked[:, j]
                    if np.any(mask):
                        b = bin_idx[mask, j]
                        w = wt[mask]
                        np.add.at(wt_bins[j], b, w)

                # Count 1->0 detachment events, attribute to pre-event ring state
                num_events = np.zeros_like(wt_bins)
                if C_all.shape[0] > 1:
                    detach_mask = (C_all[:-1] == 1) & (C_all[1:] == 0)
                    Ring1_prev = Ring1[:-1]
                    dist_prev = np.abs(Ring1_prev[:, None] - CAT_pos_all[None, :])
                    dist_prev = np.where(dist_prev > halfL, L - dist_prev, dist_prev)
                    bin_prev = np.digitize(dist_prev, bins_, right=False) - 1
                    bin_prev = np.clip(bin_prev, 0, nbins - 1)
                    for j in range(self.num_motifs):
                        idx = detach_mask[:, j]
                        if np.any(idx):
                            b = bin_prev[idx, j]
                            np.add.at(num_events[j], b, 1)

                # Rates per motif/bin
                rates = num_events / (wt_bins + 1e-10)
                rates[wt_bins <= 1e-5] = np.nan

                for j in range(self.num_motifs):
                    k_CAT_Ring_array.append(rates[j])
        k_CAT_Ring_array = np.array(k_CAT_Ring_array)
        return unique_distance,k_CAT_Ring_array
    def k_right(self,path):
        a_folder = path +"/transition_data"
        k_attach_close_array = []
        k_detach_close_array = []
        for k in range(self.num_trials):
            file = "%04d.txt"%(k+1)
            data = np.loadtxt(open(a_folder+"/"+file))
            if len(data.shape) == 2:
                wt = data[:,1]
                replica_idx = np.arange(len(wt)).reshape(-1,1)
                Ring1 = data[:,3]
                Ring1_idx = np.int32(Ring1.reshape(-1,1)/self.repeated_length)
                Ring1_idx[Ring1_idx == self.num_motifs] = 0
                Ring1_change = Ring1[1:]-Ring1[:-1]
                Ring1_not_change_idx = np.where(Ring1_change == 0)[0]
                C1 = data[:,4:8]
                C1[C1 < 0] = 0
                C1_close = C1[replica_idx,Ring1_idx]
                C1_change = C1_close[1:] - C1_close[:-1]
                C1_increase_1_idx = np.where(C1_change == -1)[0]
                C1_decrease_1_idx = np.where(C1_change == 1)[0]
                intersect_idx = reduce(np.intersect1d, (Ring1_not_change_idx,C1_increase_1_idx))
                num_increase = len(intersect_idx)
                intersect_idx = reduce(np.intersect1d, (Ring1_not_change_idx,C1_decrease_1_idx))
                num_decrease = len(intersect_idx)
                in_0 = np.where(np.all(C1_close == 0,axis = 1))[0]  # not blocked close
                time_in_0 = np.sum(wt[in_0])
                in_1 = np.where(np.all(C1_close == 1,axis = 1))[0]  # not blocked close
                time_in_1 = np.sum(wt[in_1])
                if time_in_0!= 0:
                    k_attach_close_array.append(num_increase/time_in_0)
                if time_in_1!= 0:
                    k_detach_close_array.append(num_decrease/time_in_1)
        k_attach_close_array = np.array(k_attach_close_array)
        k_detach_close_array = np.array(k_detach_close_array)
        k_attach_close_mean = np.mean(k_attach_close_array)
        k_attach_close_err = np.std(k_attach_close_array)/np.sqrt(len(k_attach_close_array))
        k_detach_close_mean = np.mean(k_detach_close_array)
        k_detach_close_err = np.std(k_detach_close_array)/np.sqrt(len(k_detach_close_array))
        return k_attach_close_mean,k_attach_close_err,k_detach_close_mean,k_detach_close_err
    def k_left(self,path):
        a_folder = path +"/transition_data"
        k_attach_array = []
        k_detach_array = []
        for k in range(self.num_trials):
            file = "%04d.txt"%(k+1)
            data = np.loadtxt(open(a_folder+"/"+file))
            if len(data.shape) == 2:
                wt = data[:,1]
                replica_idx = np.arange(len(wt)).reshape(-1,1)
                Ring1 = data[:,3]
                Ring1_idx = np.int32(Ring1.reshape(-1,1)/self.repeated_length)
                Ring1_idx[Ring1_idx == self.num_motifs] = 0
                Ring1_idx_left = Ring1_idx-1
                Ring1_idx_left[Ring1_idx_left< 0] += self.num_motifs
                Ring1_change = Ring1[1:]-Ring1[:-1]
                Ring1_not_change_idx = np.where(Ring1_change == 0)[0]
                C1 = data[:,4:8]
                C1[C1 < 0] = 0
                C1_close = C1[replica_idx,Ring1_idx_left]
                C1_change = C1_close[1:] - C1_close[:-1]
                C1_increase_1_idx = np.where(C1_change == -1)[0]
                C1_decrease_1_idx = np.where(C1_change == 1)[0]
                intersect_idx = reduce(np.intersect1d, (Ring1_not_change_idx,C1_increase_1_idx))
                num_increase = len(intersect_idx)
                intersect_idx = reduce(np.intersect1d, (Ring1_not_change_idx,C1_decrease_1_idx))
                num_decrease = len(intersect_idx)
                in_0 = np.where(np.all(C1_close == 0,axis = 1))[0]  # not blocked close
                time_in_0 = np.sum(wt[in_0])
                in_1 = np.where(np.all(C1_close == 1,axis = 1))[0]  # not blocked close
                time_in_1 = np.sum(wt[in_1])
                if time_in_0!= 0:
                    k_attach_array.append(num_increase/time_in_0)
                if time_in_1!= 0:
                    k_detach_array.append(num_decrease/time_in_1)
        k_attach_array = np.array(k_attach_array)
        k_detach_array = np.array(k_detach_array)
        k_attach_mean = np.mean(k_attach_array)
        k_attach_err = np.std(k_attach_array)/np.sqrt(len(k_attach_array))
        k_detach_mean = np.mean(k_detach_array)
        k_detach_err = np.std(k_detach_array)/np.sqrt(len(k_detach_array))
        return k_attach_mean,k_attach_err,k_detach_mean,k_detach_err
    def P_occupied(self,path):
        a_folder = path +"/transition_data"
        P_right_occupied_array = []
        P_left_occupied_array = []
        for k in range(self.num_trials):
            file = "%04d.txt"%(k+1)
            data = np.loadtxt(open(a_folder+"/"+file))
            if len(data.shape) == 2:
                wt = data[:,1]
                replica_idx = np.arange(len(wt)).reshape(-1,1)
                Ring1 = data[:,3]
                Ring1_idx = np.int32(Ring1.reshape(-1,1)/self.repeated_length)
                Ring1_idx_left = Ring1_idx-1
                Ring1_idx_left[Ring1_idx_left< 0] += self.num_motifs
                C1 = data[:,4:8]
                C1[C1 < 0] = 0
                C1_close = C1[replica_idx,Ring1_idx].reshape(1,-1)
                C1_far =  C1[replica_idx,Ring1_idx_left].reshape(1,-1)
                #print(C1_close.shape)
                P_right_occupied = np.sum(C1_close*wt)/np.sum(wt)
                P_left_occupied = np.sum(C1_far*wt)/np.sum(wt)
                P_right_occupied_array.append(P_right_occupied)
                P_left_occupied_array.append(P_left_occupied)
                #print(P_right_occupied)
        P_right_occupied_array = np.array(P_right_occupied_array)
        P_left_occupied_array = np.array(P_left_occupied_array)
        P_right_occupied_mean = np.mean(P_right_occupied_array)
        P_right_occupied_err = np.std(P_right_occupied_array)/np.sqrt(len(P_right_occupied_array))
        P_left_occupied_mean = np.mean(P_left_occupied_array)
        P_left_occupied_err = np.std(P_left_occupied_array)/np.sqrt(len(P_left_occupied_array))
        return P_right_occupied_mean,P_right_occupied_err,P_left_occupied_mean,P_left_occupied_err
    def P_occupied_ring_ahead_behind(self,path):
        a_folder = path +"/simulation_data"
        P_right_ahead_array = []
        P_right_behind_array = []
        P_left_ahead_array = []
        P_left_behind_array = []
        for k in range(self.num_trials):
            file = "%04d.txt"%(k+1)
            data = np.loadtxt(open(a_folder+"/"+file))
            if len(data.shape) == 2:
                print("ok")
        
class other_analysis(nonequilibrium_analysis):
    def __init__(self):
        super().__init__()
    # nonlinear coupling function
    def potential_FENE(self,dr,kNL = 0,n= 10,cross_distance = 10):
        delta_x = np.sqrt((dr)**2+cross_distance**2)
        U = -0.5 * kNL * delta_x**2 * np.log(1 - (delta_x/n)**2)
        return U
        
    