import torch
from numpy import pi, sin, cos
import numpy as np

"""
Correlation Functions
"""
    
def SSF_parallel(field):
    return torch.abs(torch.fft.fft2(field[2], norm="ortho"))**2

def SSF_perpendicular(field):
    return torch.sum(torch.abs(torch.fft.fft2(field[0:2], norm="ortho"))**2, dim=0)

"""
Sampling
"""


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

# Fitting

def fit_line(x, y):
    # model: y ~ N(mx + b, sigma)
    # so minimize y - ([x; 1][m; b])
    A = torch.cat((x.view(-1, 1), torch.ones((x.numel(), 1))), axis=1)
    return (torch.linalg.inv(A.T @ A) @ A.T @ y.reshape(-1, 1)).reshape(-1)
