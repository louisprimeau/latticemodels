import torch
from torch import sin, cos
from numpy import pi

def Ry(phi):
    return torch.Tensor([[cos(phi),  0,   sin(phi)],
                         [ 0,              1,   0             ],
                         [-sin(phi), 0,   cos(phi)]])

def Rz(phi):
    return torch.Tensor([[cos(phi), -sin(phi), 0],
                         [sin(phi),  cos(phi), 0],
                         [0         ,      0,              1]])

def spherical_to_cartesian(phi, theta):
    return torch.cat((cos(theta) * sin(phi),
                      sin(theta) * sin(phi),
                      cos(phi)), axis=0) # shape (3, N)

def reflect_across(x, y):
    new_v = 2 * y.dot(x) * x - y
    return new_v

def random_theta_vector(v, phi):
    us = spherical_to_cartesian(phi.view(1, -1), (torch.rand(1)*2*pi).view(1,-1))
    Ry = Ry(torch.acos(v[2]))
    Rz = Rz(torch.atan2(v[1], v[0]))
    us = Rz.mm(Ry.mm(us))
    return (us / torch.norm(us, dim=0)).T

def normalize(vector, dim=0):
    return vector / torch.norm(vector, dim=dim)

def sph_tr_area(s1, s2, s3):
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
    return (sph_tr_area(s1, s2, s3) + sph_tr_area(s1, s3, s4)) / (4 * pi)

# The topological charge density affected by the spin s.
# 
#       sn--sne
#     /6|1/ 2|
#   sw--s---se
#   |5/4|3 /
#   ssw-ss
#
def spin_tpd(s, sn, se, ss, sw, sne, ssw):
    temp = (sph_tr_area(s,   sne, sn),
            sph_tr_area(s,   se,  sne),
            sph_tr_area(ss,  se,  s ),
            sph_tr_area(ssw, ss,  s),
            sph_tr_area(ssw, s,   sw),
            sph_tr_area(sw,  s,   sn))
    return sum(temp)

def topological_density(O3field):
    # assume shape is (3, N, N)
    return local_topological_density(O3field.flatten(1).T, # x
                                     torch.roll(O3field,  1, 1).flatten(1).T, # x + xhat
                                     torch.roll(O3field,  (1,1), (1,2)).flatten(1).T, # x + xhat + yhat
                                     torch.roll(O3field, 1, 2).flatten(1).T, # x + yhat
                                     ).reshape(O3field.size(1), O3field.size(2))

def topological_charge(O3field):
    return torch.sum(topological_density(O3field))