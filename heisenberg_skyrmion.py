import torch
import matplotlib.pyplot as plt
import numpy as np

from lattice_utils import *


torch.set_default_tensor_type(torch.FloatTensor)

def H(s, st, sr, sb, sl, params):
    J, Dx, Dy, h = params
    nne = -J * (s.dot(st) + s.dot(sr) + s.dot(sb) + s.dot(sl))
    dme = -Dx.dot(s.cross(sr)) - Dy.dot(s.cross(st))
    zme = -h.dot(s)
    return nne + dme + zme
    
def partition_function(E, T, kb):
    return torch.exp(-E / (kb * T))

N = 1000
space_dim = 2
spin_dim = 3
kb = 1

J = 1
B = 0.5
Dx, Dy = J * torch.Tensor([1, 0, 0]), J * torch.Tensor([0, 1, 0]),
h = B * torch.Tensor([0, 0, 1])
params = J, Dx, Dy, h

field = get_paramagnetic_field(N, space_dim, spin_dim)

temperatures = torch.linspace(3, 0.01, 200)
annealing_steps = 10000

Q = topological_charge(field)
print("Q = {:.2f}".format(Q.item()))

for i, T in enumerate(temperatures):
    print("At temperature T={:.2f} ({} / {})".format(T.item(), i, len(temperatures)))
    n_accepted = 0
    for iteration in range(annealing_steps):
          
        idx = get_random_idx(field)
        
        neighbors = nearest_neighbor2Dperiodic(idx, field.size()[1:])
        neighbor_spins = tuple(field[:, i[0], i[1]].view(-1) for i in neighbors)

        current_spin = field[:, idx[0], idx[1]].view(-1)        
        current_energy = H(current_spin, *neighbor_spins, params)
        
        new_spin = torch.randn(field.size(0))
        new_spin = new_spin / torch.norm(new_spin)
        new_energy = H(new_spin, *neighbor_spins, params)

        if new_energy < current_energy:
            field[:, idx[0], idx[1]] = new_spin.view(-1, 1)
            n_accepted += 1
        else:
            update_prob = partition_function(new_energy - current_energy, T, kb)
            if torch.bernoulli(update_prob):
                field[:, idx[0], idx[1]] = new_spin.view(-1, 1)
                n_accepted += 1
        if (iteration+1) % 1000 == 0:
          print("\t Completed {}/{} annealing steps".format(iteration+1, annealing_steps))

    print("{}/{} steps accepted".format(n_accepted, annealing_steps))

    #S = torch.mean(field, dim=(1,2))
    #print("Average S: ({}, {}, {})".format(S[0].item(), S[1].item(), S[2].item()))

    Q = topological_charge(field)
    print("Q = {:.2f}".format(Q.item()))
    print("------------------------------")
field = field.numpy()


"""
fig = plt.figure()
ax = fig.add_subplot()

stride = 1

# Make the grid
x, y = np.meshgrid(np.linspace(0, 1, N // stride),
                   np.linspace(0, 1, N // stride))

# Make the direction data for the arrows
field2d = field[:2, ::stride, ::stride]
field2d = field2d / np.linalg.norm(field2d, axis=0)

ax.quiver(x, y, field2d[0], field2d[1])

plt.show()
"""
