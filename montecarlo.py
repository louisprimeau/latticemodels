import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from lattice_utils import *

def metropolis_epoch(field, T, H, params, partition_function, min_steps):

    iteration, n_accepted = 0, 0
    while n_accepted <= min_steps:
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

        if (iteration+1) % 10000 == 0:
            print("\t {} steps, {} accepted. \tQ = {:.3f}".format(iteration+1, n_accepted, topological_charge(field)))
        
        iteration += 1

    return field

def overrelaxation_epoch(field, H, params):
    dims = torch.meshgrid([torch.arange(i) for i in field.size()[1:]])
    indices = torch.cat([dim.reshape(1, -1) for dim in dims], axis=0)
    indices = indices[:, torch.randperm(indices.size(1))].T
    for idx in indices:
        neighbors = nearest_neighbor2Dperiodic(idx, field.size()[1:])
        neighbor_spins = tuple(field[:, i[0], i[1]].view(-1) for i in neighbors)
        current_spin = field[:, idx[0], idx[1]].view(-1)
        new_spin = or_compute_spin(current_spin, *neighbor_spins, params)
        field[:, idx[0], idx[1]] = new_spin.view(-1)
    return field

def metropolis(field, H, params, partition_function, temperatures, min_steps):

    for i, T in enumerate(temperatures):
        start_time = time.perf_counter()
        print("At temperature T={:.2f} ({} / {})".format(T.item(), i+1, len(temperatures)))
        metropolis_epoch(field, T, H, params, partition_function, min_steps)
        print("Took {:.1f} s".format(time.perf_counter() - start_time))
        print("------------------------------")

    return field

def overrelaxation(field, H, params, partition_function, temperatures, min_steps):

    for i, T in enumerate(temperatures):
        start_time = time.perf_counter()

        print("At temperature T={:.2f} ({} / {})".format(T.item(), i+1, len(temperatures)))

        n = 4
        for i in range(1, n+1):
            field = metropolis_epoch(field, T, H, params, partition_function, min_steps // i)
            print("Metropolis Epoch finished.")
            field = overrelaxation_epoch(field, H, params)
            print("Overrelaxation Epoch finished.")
        
        print("Took {:.1f} s".format(time.perf_counter() - start_time))
        print("------------------------------")
        
    return field

def microcanonical_epoch(field, H, params):

    charge_limit = 10 * torch.mean(torch.abs(topological_density(field)))
    demon_charge = torch.Tensor([0.0])
    demon_energy = torch.Tensor([0.0])
    
    for i in range(field[0, :].numel()):
        idx = get_random_idx(field)
        neighbors = seven_neighbors2Dperiodic(idx, field.size()[1:])
        neighbor_spins = tuple(field[:, i[0], i[1]].view(-1) for i in neighbors)
        
        old_spin = field[:, idx[0], idx[1]].view(-1)
        field_axis = min_energy_vector(*neighbor_spins[:4], params)
        field_axis = field_axis / torch.norm(field_axis)
        angle = torch.acos(old_spin.dot(field_axis))
        newspins = constant_energy_vectors(field_axis, angle, 10)

        newEs = H_batched(newspins, *neighbor_spins[:4], params)
        oldQ = sevenpoint_topologicaldensity(old_spin, *neighbor_spins)
        newQs = sevenpoint_topologicaldensity(newspins, *neighbor_spins)

        
        #new_spin = random_S2v()
        #oldQ = sevenpoint_topologicaldensity(current_spin, *neighbor_spins)
        #newQ = sevenpoint_topologicaldensity(new_spin, *neighbor_spins)    
        #oldE = H(current_spin, *neighbor_spins[:4], params)
        #newE = H(new_spin, *neighbor_spins[:4], params)
        
    return field


def microcanonical_demon(field, H, params, partition_function):
    field = microcanonical_epoch(field, H, params)
    return field

def heatbath_epoch(field, T, H, params, partition_function, steps):
    sampler = HeatbathSampler()
    for step in range(steps):
        idx = get_random_idx(field)
        neighbors = nearest_neighbor2Dperiodic(idx, field.size()[1:])
        neighbor_spins = tuple(field[:, i[0], i[1]].view(-1) for i in neighbors)
        field_vector = min_energy_vector(*neighbor_spins[:4], params)
        new_spin = sampler.one_sample(field_vector, torch.norm(field_vector)/T)
        new_spin = new_spin / torch.norm(new_spin)
        field[:, idx[0], idx[1]] = new_spin.view(-1, 1)
        
    return field

def heatbath(field, H, params, partition_function, temperatures, steps):
    print("Heatbath Algorithm:")
    for i, T in enumerate(temperatures):
        start_time = time.perf_counter()
        print("At temperature T={:.2f} ({} / {})".format(T.item(), i+1, len(temperatures)))
        field = heatbath_epoch(field, T, H, params, partition_function, steps)
        print("Took {:.1f} s".format(time.perf_counter() - start_time))
        print("------------------------------")
    return field
