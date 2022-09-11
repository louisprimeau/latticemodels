import torch

"""
Functions for manipulating fields of S2 Spins
on periodic square lattices.
"""

#            dim[1]
#        o------------> +y
#        |
#dim[0]  |
#        |
#        v
#       +x


def get_random_binary_field(N, dim=2):
    return torch.bernoulli(torch.ones((N,)*dim)*0.5)

def get_random_normal_field(N, space_dim=2, spin_dim=3):
    arr = torch.empty([spin_dim] + [N]*space_dim).normal_(mean=0, std=1)
    return arr / torch.norm(arr, dim=0)

def get_Z_field(N, space_dim=2, spin_dim=3):
    arr = torch.zeros([spin_dim] + [N]*space_dim)
    arr[2, :, :] = torch.ones(N,N)
    return arr

#      O ----- X ----- O
#      |       |       |
#      |       |       |
#      X ----- X ----- X
#      |       |       |
#      |       |       |
#      O ----- X ----- O

def nearest_neighbor2Dperiodic(coordinate, dims):
    top = coordinate[0], (coordinate[1] + 1) % dims[1]
    bot = coordinate[0], (coordinate[1] - 1)# % dims[1]
    rig = (coordinate[0] + 1) % dims[1], coordinate[1]
    lef = (coordinate[0] - 1), coordinate[1]
    return top, rig, bot, lef

#      O ----- X ----- X
#      |       |       |
#      |       |       |
#      X ----- X ----- X
#      |       |       |
#      |       |       |
#      X ----- X ----- O

def seven_neighbors2Dperiodic(coordinate, dims):
    north = coordinate[0], (coordinate[1] + 1) % dims[1]
    east = (coordinate[0] + 1) % dims[1], coordinate[1]
    south = coordinate[0], (coordinate[1] - 1)
    west = (coordinate[0] - 1), coordinate[1]
    northeast = (coordinate[0] + 1) % dims[0], (coordinate[1] + 1) % dims[1]
    southwest = (coordinate[0] - 1), coordinate[1] - 1
    return north, east, south, west, northeast, southwest
