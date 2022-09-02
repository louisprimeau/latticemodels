import torch
from numpy import pi, sin, cos
import numpy as np
import matplotlib.pyplot as plt
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


"""
Hamiltonian: Exchange + DM + Zeeman interactions
"""

def reflect_across(x, y):
    new_v = 2 * y.dot(x) * x - y
    return new_v

def min_energy_vector(st, sr, sb, sl, params):
    J, Dx, Dy, h = params
    min_vector = J * (st + sr + sb + sl) \
                 + sr.cross(Dx) + Dx.cross(sl) \
                 - st.cross(Dy) - Dy.cross(sb) \
                 + h
    return min_vector

def constant_energy_vectors(v, phi, N):
    
    theta = torch.linspace(0, 2*np.pi, N+1)[:N].view(1, -1)
    phi = phi.repeat(N).view(1, -1)
    us = torch.cat((torch.cos(theta) * torch.sin(phi),
                    torch.sin(theta) * torch.sin(phi),
                    torch.cos(phi)), axis=0) # shape (3, N)
    vtheta = torch.atan2(v[1], v[0])
    vphi = torch.acos(v[2])
    Rx = torch.Tensor([[1, 0,           0         ],
                       [0, cos(vphi),   -sin(vphi)  ],
                       [0, sin(vphi),    cos(vphi)  ]])
    Rz = torch.Tensor([[cos(vtheta), -sin(vtheta), 0],
                       [sin(vtheta),  cos(vtheta), 0],
                       [0         ,  0         , 1]])
    us = Rz.mm(Rx.mm(us))
    return (us / torch.norm(us, dim=0)).T                    


def or_compute_spin(s, st, sr, sb, sl, params):
    min_vector = min_energy_vector(st, sr, sb, sl, params)
    min_vector = min_vector / torch.norm(min_vector)
    s = s / torch.norm(s)
    new_s = reflect_across(s, min_vector)
    return new_s

def H(s, st, sr, sb, sl, params):
    J, Dx, Dy, h = params
    nne = J * s.dot(st + sr + sb + sl)
    dme = s.dot(sr.cross(Dx) + Dx.cross(sl) - st.cross(Dy) + Dy.cross(sb))
    zme = s.dot(h)
    return - nne - dme - zme    

def H_batched(spin, st, sr, sb, sl, params):
    J, Dx, Dy, h = params
    nne = J * (spin * (st + sr + sb + sl)).sum(1)
    dme = (spin * (sr.cross(Dx) + Dx.cross(sl) - st.cross(Dy) + Dy.cross(sb))).sum(1)
    zme = (spin * h).sum(1)
    return - nne - dme - zme

"""
Sampling
"""

def get_random_idx(field):
    dims = field.size()[1:]
    return [torch.randint(low=0, high=dim, size=(1,)) for dim in dims]

def random_S2v():
    S = torch.randn(3)
    return S / torch.norm(S)

# not a uniform distribution
def spherical_cap(v, max_angle):
    theta = np.random.rand() * 2 * pi
    phi = np.random.rand() * 2 * max_angle - max_angle
    Rx = torch.Tensor([[1, 0,           0         ],
                       [0, cos(phi),   -sin(phi)  ],
                       [0, sin(phi),    cos(phi)  ]])
    Rz = torch.Tensor([[cos(theta), -sin(theta), 0],
                       [sin(theta),  cos(theta), 0],
                       [0         ,  0         , 1]])
    return Rz.mm(Rx.mm(v.view(-1, 1))).view(*v.size())

# Here we implement the heat bath distribution
# The pdf of s is determined by the pdf of its angles.
# p(phi) = B|n|exp(-B|n|cos(phi)) / (exp(B|n|) - exp(-B|n|)) over [0, pi)
# p(theta) = U([0, 2pi))
# here |n| is the magnitude of the local mean field
# and B is the thermodynamic beta = 1/kbT
# simple importance sampler
class AngularSampler():
    def __init__(self):
        return None
    def one_sample(self, u):
        max_height = self.distribution(torch.zeros(1), u)
        for i in range(10000):
            sample = [torch.rand(1) * pi, torch.rand(1) * max_height]
            if sample[1] < self.distribution(sample[0], u):
                return sample[0]            
        assert False, "Failed to sample after 10000 tries, distribution is too singular, u={}".format(u.item())

    def distribution(self, phi, u):
        return u * torch.sin(phi) * torch.exp(u * torch.cos(phi)) / (torch.exp(u) - torch.exp(-u))
        
class HeatbathSampler():
    def __init__(self):
        self.sampler = AngularSampler()
        return None
    def one_sample(self, v, u):
        try:
            phi = self.sampler.one_sample(u)
        except AssertionError:
            print(v)
            assert False
        theta = torch.rand(1) * 2 * pi
        Ry = torch.Tensor([[ cos(phi), 0, sin(phi)],
                           [ 0,        1, 0       ],
                           [-sin(phi), 0, cos(phi)]])
        Rz = torch.Tensor([[cos(theta), -sin(theta), 0],
                           [sin(theta),  cos(theta), 0],
                           [0         ,  0         , 1]])
        v = Rz.mm(Ry.mm(v.view(-1, 1)))
        return v

    
"""
Functions for Topological Charge
"""

def spherical_triangle_area(s1, s2, s3):
    assert len(s1.size()) == 2 and s1.size(1) == 3
    # assume s has shape (N, 3)
    return 2 * torch.atan2(
        torch.sum(s1 * s2.cross(s3, dim=-1), dim=1),
        1 + torch.sum(s1*s2, dim=1) + torch.sum(s2*s3, dim=1) + torch.sum(s3*s1, dim=1),
    )

# Topological charge density at the point in the dual lattice
# above and to the right of s1.
# 
#  s4---s3
#  |  /  |
#  s1---s2

def local_topological_density(s1, s2, s3, s4):
    return (spherical_triangle_area(s1, s2, s3) + spherical_triangle_area(s1, s3, s4)) / (4 * pi)

# The topological charge density affected by the spin s.
# 
#       sn--sne
#     /6|1/ 2|
#   sw--s---se
#   |5/4|3 /
#   ssw-ss
#
def sevenpoint_topologicaldensity(s, sn, se, ss, sw, sne, ssw):
    return  spherical_triangle_area(s,   sne, sn)  \
          + spherical_triangle_area(s,   se,  sne) \
          + spherical_triangle_area(ss,  se,  s )  \
          + spherical_triangle_area(ssw, ss,  s)   \
          + spherical_triangle_area(ssw, s,   sw)  \
          + spherical_triangle_area(sw,  s,  sn)

def topological_density(O3field):
    # assume shape is (3, N, N)
    return local_topological_density(torch.roll(O3field, -1, 1).flatten(1).T, # top
                                     torch.roll(O3field, 1, 2).flatten(1).T, # right
                                     torch.roll(O3field, 1, 1).flatten(1).T, # bottom
                                     torch.roll(O3field, -1, 1).flatten(1).T, # left
                                     ).reshape(O3field.size(1), O3field.size(2))

def topological_charge(O3field):
    return torch.sum(topological_density(O3field))

# Spin Structure Factor

def SSF_parallel(field):
    return torch.abs(torch.fft.fft2(field[2], norm="ortho"))**2

def SSF_perpendicular(field):
    return torch.sum(torch.abs(torch.fft.fft2(field[0:2], norm="ortho"))**2, dim=0)

# Sample Configurations

def simple_skyrmion2d(N, n):
    x, y = torch.meshgrid(torch.linspace(-1, 1, N), torch.linspace(-1, 1, N), indexing='ij')
    u = torch.sqrt(x**2 + y**2).unsqueeze(0)
    phi = torch.atan2(y, x).unsqueeze(0)
    f = lambda r: pi * (1 - r)
    S = torch.cat((torch.sin(f(u)) * torch.cos(n * phi),
                   torch.sin(f(u)) * torch.sin(n * phi),
                   torch.cos(f(u))
                   ), axis=0)
    return S
