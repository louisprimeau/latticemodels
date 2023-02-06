import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from utils.lattice_utils import *
from utils.sampling import *
from utils.lattice_geometry import *
def metropolis_epoch(field, T, H, params, partition_function, min_passes):

    system_size = field.size(1) * field.size(2)
    iteration, n_accepted = 0, 0
    correlation = torch.zeros(system_size)
    field0 = torch.clone(field)
    
    while n_accepted <= min_passes * system_size:

        idx = get_random_idx(field)
        neighbors = nearest_neighbor2Dperiodic(idx, field.size()[1:])
        neighbor_spins = tuple(field[:, i[0], i[1]].view(-1) for i in neighbors)
    
        current_spin = field[:, idx[0], idx[1]].view(-1)
        new_spin = random_S2v() #spherical_cap(current_spin, np.pi / 5)
        
        current_energy = H(current_spin, *neighbor_spins, params)
        new_energy = H(new_spin, *neighbor_spins, params)
        dE = new_energy - current_energy
        
        if new_energy < current_energy or torch.bernoulli(partition_function(dE, T)):
            field[:, idx[0], idx[1]] = new_spin.view(-1, 1)
            n_accepted += 1
            #correlation[n_accepted % system_size] = torch.mean(thermal_average(field0, field, T, params))

        if n_accepted % system_size == 0 & n_accepted > 0:
            print("\t {} steps, {} accepted. \tQ = {:.3f}".format(iteration+1, n_accepted, topological_charge(field)))
            #relaxation_time = fit_line(torch.arange(system_size), torch.log(correlation))[0]
            #print("Autocorrelation Time: {}".format(relaxation_time.item()))
            
        iteration += 1

    return field

def metropolis(field, H, params, partition_function, temperatures, min_steps):

    for i, T in enumerate(temperatures):
        start_time = time.perf_counter()
        print("At temperature T={:.2f} ({} / {})".format(T.item(), i+1, len(temperatures)))
        metropolis_epoch(field, T, H, params, partition_function, min_steps)
        print("Took {:.1f} s".format(time.perf_counter() - start_time))
        print("------------------------------")

    return field
