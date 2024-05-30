from typing import Any

import torch
from torch import nn
import pandas as pd


class CIR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.k = nn.Parameter(torch.randn(1))
        self.theta = nn.Parameter(torch.randn(1))

    def forward(self, r, dt, sigma, *, epsilon=1, sample=False):
        ode_part = self.k * (self.theta - r) * dt
        sde_part = sigma * torch.sqrt(r * dt) * epsilon
        if sample:
            sde_part = sde_part * torch.randn(1)
        return ode_part + sde_part, 2 * self.k * self.theta - sigma ** 2
    
class StepLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, r_predict, r_true, constraint):
        return (r_predict - r_true) ** 2 #nn.ReLU()(-constraint)