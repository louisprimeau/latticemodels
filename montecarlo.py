import torch
import time

from lattice_utils import *

def metropolis(field, H, partition_function, temperatures, annealing_steps):

    for i, T in enumerate(temperatures):
        start_time = time.time()
        print("At temperature T={:.2f} ({} / {})".format(T.item(), i+1, len(temperatures)))
        iteration, n_accepted = -1, 0
        while True:
            iteration += 1
            if iteration >= annealing_steps and n_accepted >= 10000:
                break
            
            idx = get_random_idx(field)

            neighbors = nearest_neighbor2Dperiodic(idx, field.size()[1:])
            neighbor_spins = tuple(field[:, i[0], i[1]].view(-1) for i in neighbors)

            current_spin = field[:, idx[0], idx[1]].view(-1)        
            current_energy = H(current_spin, *neighbor_spins)

            new_spin = torch.randn(field.size(0))
            new_spin = new_spin / torch.norm(new_spin)
            new_energy = H(new_spin, *neighbor_spins)

            if new_energy < current_energy:
                field[:, idx[0], idx[1]] = new_spin.view(-1, 1)
                n_accepted += 1
            else:
                update_prob = partition_function(new_energy - current_energy, T)
                if torch.bernoulli(update_prob):
                    field[:, idx[0], idx[1]] = new_spin.view(-1, 1)
                    n_accepted += 1
            if (iteration+1) % 10000 == 0:
                print("\t Completed {}/{} annealing steps".format(iteration+1, annealing_steps))

        print("{}/{} steps accepted".format(n_accepted, iteration))
        
        Q = topological_charge(field)

        print("Q = {:.2f}".format(Q.item()))
        print("Took {:.1f} s".format(time.time() - start_time))
        print("------------------------------")

    return field
