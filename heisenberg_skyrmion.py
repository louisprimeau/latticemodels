import torch
import matplotlib.pyplot as plt

from utils.lattice_utils import *
from utils.lattice_geometry import get_random_normal_field
from evolution.metropolis import metropolis
from evolution.heatbath import heatbath
from utils.hamiltonian import boltzmann
from utils.hamiltonian import H

torch.set_default_tensor_type(torch.DoubleTensor)

N = 20
space_dim = 2
spin_dim = 3

J, B, D = 1, 1, 2
Dx, Dy = D * torch.Tensor([1, 0, 0]), D * torch.Tensor([0, 1, 0]),
h = B * torch.Tensor([0, 0, 1])
params = J, Dx, Dy, h

field = get_random_normal_field(N, space_dim, spin_dim)
#field = simple_skyrmion2d(N, 1)
temperatures = torch.linspace(3, 0.01, 50)
annealing_passes = 200
oldfield = torch.clone(field)

#field = metropolis(field, H, params, boltzmann, temperatures, annealing_passes)
field = heatbath(field, H, params, boltzmann, temperatures, annealing_passes*1000)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(oldfield[2].numpy(), vmin=-1, vmax=1, cmap='jet')
ax2.imshow(field[2].numpy(), vmin=-1, vmax=1, cmap='jet')

plt.show()
