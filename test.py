from misc.wald_utilities import MTF
import numpy as np
import torch
from config import bargs
from MTF import wald_protocol

d = []
for _ in range(100):
    a = torch.randn(1, 8, 64, 64).to(bargs.device, bargs._dtype)
    b = MTF(a, bargs.sensor, 1, bargs.channel)
    c = wald_protocol(a, 1)
    d.append(torch.sum(c - b))

print(torch.mean(torch.Tensor(d)))

# h = np.array([-3 + 3j, -4 + 8j, 1 + 1j, 3 - 3j])
# print(np.max(h))
# h = np.clip(h, a_min=0, a_max=np.max(h))
# print(h)
