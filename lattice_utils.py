import torch
from numpy import pi

def get_random_binary_field(N, dim=2):
    return torch.bernoulli(torch.ones((N,)*dim)*0.5)

def get_random_normal_field(N, space_dim=2, spin_dim=3):
    arr = torch.empty([spin_dim] + [N]*space_dim).normal_(mean=0, std=1)
    return arr / torch.norm(arr, dim=0)

def get_paramagnetic_field(N, space_dim=2, spin_dim=3):
    arr = torch.zeros([spin_dim] + [N]*space_dim)
    arr[2, :, :] = torch.ones(N,N)
    return arr

def _nearest_neighbor(coordinate):
    c_ups = [coordinate[:] for _ in range(len(coordinate))]
    c_downs = [coordinate[:] for _ in range(len(coordinate))]
    for i in range(len(coordinate)):
        c_ups[i][i] += 1
        c_downs[i][i] -= 1
    return c_ups + c_downs
        
def nearest_neighbor_periodic(coordinate, dims):
    coords = _nearest_neighbor(coordinate)
    for coord in coords:
        for idx in range(len(coord)):
            if coord[idx] >= dims[idx] or coord[idx] < 0: 
                coord[idx] = coord[idx] % dims[idx]
    return coords        

def nearest_neighbor2Dperiodic(coordinate, dims):
    top = coordinate[0], (coordinate[1] + 1) % dims[1]
    bot = coordinate[0], (coordinate[1] - 1) % dims[1]
    rig = (coordinate[0] + 1) % dims[0], coordinate[1]
    lef = (coordinate[0] - 1) % dims[0], coordinate[1]
    return top, rig, bot, lef

def get_random_idx(field):
    dims = field.size()[1:]
    return [torch.randint(low=0, high=dim, size=(1,)) for dim in dims]


# Functions for topological charge

def spherical_triangle_area(s1, s2, s3):
    assert len(s1.size()) == 2 and s1.size(1) == 3
    # assume s has shape (N, 3)
    return 2 * torch.atan2(
        torch.sum(s1 * s2.cross(s3, dim=-1), dim=1),
        1 + torch.sum(s1*s2, dim=1) + torch.sum(s2*s3, dim=1) + torch.sum(s3*s1, dim=1),
    )

def spherical_triangle_area2(s1, s2, s3):
    assert len(s1.size()) == 2 and s1.size(1) == 3
    return torch.sqrt(2*(1 + torch.sum(s1*s2, dim=1))*(1 + torch.sum(s2*s3, dim=1))*(1 + torch.sum(s3*s1, dim=1)))

def local_topological_density(s1, s2, s3, s4):
    return (spherical_triangle_area2(s1, s2, s3) + spherical_triangle_area2(s1, s3, s4)) / (4 * pi)

def topological_density(O3field):
    # assume shape is (3, N, N)
    return local_topological_density(torch.roll(O3field, -1, 1).flatten(1).T, # top
                                     torch.roll(O3field, 1, 2).flatten(1).T, # right
                                     torch.roll(O3field, 1, 1).flatten(1).T, # bottom
                                     torch.roll(O3field, -1, 1).flatten(1).T, # left
                                     ).reshape(O3field.shape(0), O3field.shape(1))

def topological_charge(O3field):
    return torch.sum(topological_density(O3field))


def simple_skyrmion2d(N, n):
    x, y = torch.meshgrid(torch.linspace(-1, 1, N), torch.linspace(-1, 1, N), indexing='xy')
    u = torch.sqrt(x**2 + y**2).unsqueeze(0)
    phi = torch.atan2(y, x).unsqueeze(0)
    f = lambda r: pi * (1 - r)
    S = torch.cat((torch.sin(f(u)) * torch.cos(n * phi),
                   torch.sin(f(u)) * torch.sin(n * phi),
                   torch.cos(f(u))
                   ), axis=0)
    return S
