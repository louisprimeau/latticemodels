
import torch
import numpy as np
from numpy import pi
import vector_geometry as vg

def get_random_idx(field):
    dims = field.size()[1:]
    return [torch.randint(low=0, high=dim, size=(1,)) for dim in dims]

def random_S2v():
    S = torch.randn(3)
    return S / torch.norm(S)

# not a uniform distribution
def spherical_cap(v, max_angle):
    theta = torch.Tensor([np.random.rand() * 2 * pi])
    phi = torch.Tensor([np.random.rand() * 2 * max_angle - max_angle])
    Ry = vg.Ry(phi)
    Rz = vg.Rz(theta)
    return Rz.mm(Ry.mm(v.view(-1, 1))).view(*v.size())

class HeatbathSampler():
    def __init__(self): return None
    def one_sample(self, B, n):
        x = torch.rand(1)
        return self.invcdf_numerical(x, B, n).view(-1)
    def cdf(self, x, B, n):
        return (1 - torch.exp(-B*x - B*n)) / (1 - torch.exp(-2*B*n*torch.ones(1)))
    def invcdf_numerical(self, x, B, n):
        N = max(20*B*n, 20) #heuristic
        Y = torch.linspace(-n, n, int(N))
        X = self.cdf(Y, B, n)
        _, idx = torch.min(X <= x, 0)
        out = Y[idx-1] + (x - X[idx-1]) * (Y[idx] - Y[idx-1]) / (X[idx] - X[idx-1])
        return out
    def invcdf(self, y, B, n):
        return -torch.log(1 - y * (1 - torch.exp(-2*B*n*torch.ones(1))))/B - n

class HeatbathVectorSampler():
    def __init__(self):
        self.sampler = HeatbathSampler()
        return None
    def one_sample(self, v, B):
        n = torch.norm(v)
        energy = -self.sampler.one_sample(B, n)
        phi = torch.acos(energy / n)
        theta = torch.rand(1) * 2 * pi
        Ry = vg.Ry(phi)
        Rz = vg.Rz(theta)
        v = Rz.mm(Ry.mm(v.view(-1, 1)))
        return v / torch.norm(v) 