import torch
import matplotlib.pyplot as plt
import numpy as np
import time

from lattice_utils import *
from montecarlo import *

torch.set_default_tensor_type(torch.FloatTensor)

def H(s, st, sr, sb, sl, params):
    J, Dx, Dy, h = params
    nne = -J * s.dot(st + sr + sb + sl)
    dme = - Dx.dot(s.cross(sr) + sl.cross(s)) \
          - Dy.dot(s.cross(st) + sb.cross(s))
    zme = -h.dot(s)
    return nne + dme + zme
    
def partition_function(dE, T, kb):
    return torch.exp(-dE / (kb * T))

N = 48
space_dim = 2
spin_dim = 3
kb = 1

J = 1
B = 0.02
D = 0.2
Dx, Dy = D * torch.Tensor([1, 0, 0]), D * torch.Tensor([0, 1, 0]),
h = B * torch.Tensor([0, 0, 1])
params = J, Dx, Dy, h

field = get_random_normal_field(N, space_dim, spin_dim)
temperatures = torch.linspace(3, 0.02, 50)
annealing_steps = 100000

field = metropolis(field,
                   lambda *args: H(*args, params),
                   lambda *args: partition_function(*args, kb),
                   temperatures,
                   annealing_steps)

fig, ax = plt.subplots(1)
ax.imshow(field[2].numpy(), vmin=-1, vmax=1, cmap='jet')
plt.show()
