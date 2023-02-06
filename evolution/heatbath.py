import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from utils.sampling import HeatbathVectorSampler, get_random_idx
from utils.lattice_geometry import nearest_neighbor2Dperiodic
from utils.hamiltonian import min_energy_vector

def heatbath_epoch(field, T, H, params, partition_function, steps):
    sampler = HeatbathVectorSampler()
    for step in range(steps):

        idx = get_random_idx(field)
        old_spin = field[:, idx[0], idx[1]]

        neighbors = nearest_neighbor2Dperiodic(idx, field.size()[1:])
        neighbor_spins = tuple(field[:, i[0], i[1]].view(-1) for i in neighbors)
        field_vector = min_energy_vector(*neighbor_spins, params)

        new_spin = sampler.one_sample(field_vector, 1 / T)
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
