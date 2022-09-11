import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from utils.sampling import get_random_idx
from utils.lattice_geometry import seven_neighbors2Dperiodic
from utils.hamiltonian import min_energy_vector, H_batched

def microcanonical_epoch(field, H, params):

    charges = []
    for i in range(10000):
        idx = get_random_idx(field)
        neighbors = seven_neighbors2Dperiodic(idx, field.size()[1:])
        ns13 = tuple(field[:, ij[0], ij[1]].view(1, 3) for ij in neighbors)
        ns3 = tuple(field[:, ij[0], ij[1]].view(3) for ij in neighbors)
        field_axis = min_energy_vector(*ns3[:4], params)
        field_axis = field_axis / torch.norm(field_axis)
        
        oldspin = field[:, idx[0], idx[1]].view(1, 3)
        oldE = H_batched(oldspin, *ns3[:4], params)

        angle = torch.acos(oldspin.view(-1).dot(field_axis))
        oldfield = torch.clone(field)
        newspin = random_theta_vector(field_axis, angle).view(1, 3)
        newE = H_batched(newspin, *ns3[:4], params)
        dQ = spin_tpd(oldspin, *ns13) - spin_tpd(newspin, *ns13)
        old_topological_charge = topological_charge(field)
        if dQ > 0.001: continue
        field[:, idx[0], idx[1]] = newspin.view(3, 1)
        new_topological_charge = topological_charge(field)
        print(dQ, new_topological_charge - old_topological_charge)
        if not torch.allclose(new_topological_charge, old_topological_charge):
            breakpoint()
        charges.append(topological_charge(field))
    
    return field, charges


def microcanonical(field, H, params, partition_function):
    field = microcanonical_epoch(field, H, params)
    return field
