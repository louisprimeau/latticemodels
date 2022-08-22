import torch

def get_random_binary_field(N, dim=2):
    return torch.bernoulli(torch.ones((N,)*dim)*0.5)

def get_random_normal_field(N, space_dim=2, spin_dim=3):
    arr = torch.empty([spin_dim] + [space_dim]*2).normal_(mean=0, std=1)
    return arr / torch.norm(arr, dim=0)

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
