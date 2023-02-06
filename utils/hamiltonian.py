import torch
from . import vector_geometry as vg
from numpy import pi

def boltzmann(E, T): 
    return torch.exp(-E / T)

"""
Hamiltonian: Exchange + DM + Zeeman interactions
"""

# Local Expressions

# st, sr, sb, sl shape (N, 3)
def min_energy_vector(st, sr, sb, sl, params):
    J, Dx, Dy, h = params
    min_vector = J * (st + sr + sb + sl) \
                 + sr.cross(Dx) + Dx.cross(sl) \
                 - st.cross(Dy) + Dy.cross(sb) \
                 + h
    return min_vector

def H(spin, st, sr, sb, sl, params): # size(3)
    return -spin.dot(min_energy_vector(st, sr, sb, sl, params))

def H_batched(spin, st, sr, sb, sl, params): # size(N,3)
    return -(spin * min_energy_vector(st, sr, sb, sl, params)).sum(1)

# Whole lattice expressions
def min_energy_lattice(field, params):
    st = torch.roll(field, 1, 2).transpose(0, 2)
    sb = torch.roll(field, -1, 2).transpose(0, 2)
    sr = torch.roll(field, 1, 1).transpose(0, 2)
    sl = torch.roll(field, -1, 1).transpose(0, 2)
    return min_energy_vector(st, sr, sb, sl, params).transpose(0, 2)

def H_lattice(field, params): # size (3, N, N)
    f = min_energy_lattice(field, params)
    return (field * f).sum(0)

def total_energy(field, params):
    return torch.sum(H_lattice(field, params)) / 2

def or_compute_spin(s, st, sr, sb, sl, params):
    min_vector = vg.normalize(min_energy_vector(st, sr, sb, sl, params))
    return vg.reflect_across(vg.normalize(s), min_vector)

