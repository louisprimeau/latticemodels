import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from utils.lattice_geometry import nearest_neighbor2Dperiodic
from .metropolis import metropolis_epoch

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
