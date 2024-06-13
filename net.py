from typing import Any, Tuple, Iterable

import torch
from torch import nn
import pandas as pd

def sliding_window(ite, size=1, step=1):
    """
    Utility Function to get n size window slice of an iterable object in n step
    e.g. [1,2,3,4] -> [[1,2], [2,3], [3,4]]
    """
    return [ite[i:i+size] for i in range(0, len(ite) - size + 1, step)]
    
class CIRCell(nn.Module):
    """
    Network Cell to calculate prediction for one step

    The SDE Iterative Formula of swaption rates for CIR is
    r_{t_(t+1)} = r_t + k * (\theta - r_t) * dt + \sigma * \sqrt{r_t * dt} * \epsilon

    where \sigma is a regression prediction result of \sigma = \beta_0 + \beta_1 * \sigma_1 + ... + \beta_n * \sigma_n
    and \epsilon is a SMC(Particle Method) result of \epsilon = w_1 * \epsilon_1 + w_2 * \epsilon_2 + ... + w_n * \epsilon_n 
    """
    def __init__(self, 
                 sigma_num:int, 
                 epsilon_num:int, 
                 k:torch.Tensor=torch.randn(1), 
                 theta:torch.Tensor=torch.randn(1)) -> None:
        super().__init__()
        self.sigma_num:int = sigma_num
        self.epsilon_num:int = epsilon_num

        self.sigma_linear_layer = nn.Linear(self.sigma_num, 1)
        self.epsilon_linear_layer = nn.Linear(self.epsilon_num, 1, bias=False)
        self.softmax = nn.Softmax()

        self.k = nn.Parameter(k)
        self.theta = nn.Parameter(theta)
    
    def forward(self, r, dt, sigmas, epsilons):
        sigma = self.sigma_linear_layer(sigmas)
        epsilon = self.epsilon_linear_layer(epsilons)

        ode_part = r + self.k * (self.theta - r) * dt
        sde_part = sigma * torch.sqrt(torch.abs(r * dt)) * epsilon

        r_next = ode_part + sde_part
        reg = 2 * self.k * self.theta - sigma ** 2

        return r_next, reg
    
    def info(self):
        info_dict = {
            "k":self.k.item(),
            "theta":self.theta.item(),
            "sigmas_weights":self.sigma_linear_layer.weight.detach().tolist(),
            "sigmas_bias":self.sigma_linear_layer.bias.item(),
            "epsilons_weights":self.epsilon_linear_layer.weight.detach().tolist()
        }
        return info_dict
    
class CIRNet(nn.Module):
    """
    Network to calculate predictions for one trace (from initial step to the end)

    The SDE Iterative Formula of swaption rates for CIR is
    r_{t_(t+1)} = r_t + k * (\theta - r_t) * dt + \sigma * \sqrt{r_t * dt} * \epsilon

    where \sigma is a regression prediction result of \sigma = \beta_0 + \beta_1 * \sigma_1 + ... + \beta_n * \sigma_n
    and \epsilon is a SMC(Particle Method) result of \epsilon = w_1 * \epsilon_1 + w_2 * \epsilon_2 + ... + w_n * \epsilon_n 
    """
    def __init__(self, 
                 sigma_num:int, 
                 epsilon_num:int, 
                 k:torch.Tensor=torch.randn(1), 
                 theta:torch.Tensor=torch.randn(1)) -> None:
        super().__init__()

        self.sigma_num:int = sigma_num
        self.epsilon_num:int = epsilon_num

        self.cir_cell = CIRCell(self.sigma_num, self.epsilon_num, k, theta)

    def forward(self, trace_data)->Tuple[torch.Tensor]:
        r_predict = trace_data[0][1]
        r_predicts = []
        regs = []
        dts = []
        for step_data, step_data_next in sliding_window(trace_data, size=2):
            t = step_data[0]
            t_next = step_data_next[0]
            dt = t_next - t
            dts.append(dt)

            r = r_predict
            sigmas = step_data[2:2+self.sigma_num].view(self.sigma_num)
            epsilons = step_data[2+self.sigma_num:2+self.sigma_num+self.epsilon_num].view(self.epsilon_num)

            r_predict, reg = self.cir_cell(r, dt, sigmas, epsilons)
            r_predicts.append(r_predict)
            regs.append(reg)
        return torch.cat(r_predicts), torch.cat(regs), torch.stack(dts)
    
    def info(self):
        return self.cir_cell.info()


class StepLossCell(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, r_predict, r_true, reg):
        return (r_predict - r_true) ** 2 + nn.ReLU()(-reg)

class TraceLoss(nn.Module):
    """
    For each trace, the optimization loss function is:
        min E[w(t) * ((r_predict_t - r_true_t )^2 + reg(t))]
    And regularization condition is that punish when
        2 * k * \theta < \sigma ^ 2
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, r_predicts, regs, trace_data:torch.Tensor, weights:torch.Tensor=None):
        r_trues = trace_data.transpose(0,1)[1][1:]
        no_weighted_loss = (r_predicts - r_trues).square() + nn.ReLU()(-regs)

        if weights is not None:
            weighted_loss = no_weighted_loss * weights
            return weighted_loss.sum()
        else:
            return no_weighted_loss.sum()