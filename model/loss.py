#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
from scipy import stats

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MTL loss (Base-class)
class MTLLoss(nn.Module):
    def __init__(self, loss_reg, loss_clsf, strategy):
        super().__init__()
        self.loss_reg = loss_reg
        self.loss_clsf = loss_clsf
        self.strategy = strategy

    def _is_rainrate_epoch(self, epoch):
        if epoch >= self.strategy["rainrate_start"] and epoch < self.strategy["rainrate_end"]:
            return True
        else:
            return False
    
    def _is_rainmask_epoch(self, epoch):
        if epoch >= self.strategy["rainmask_start"] and epoch < self.strategy["rainmask_end"]:
            return True
        else:
            return False
    
    def _is_cloudwater_epoch(self, epoch):
        if epoch >= self.strategy["cloudwater_start"] and epoch < self.strategy["cloudwater_end"]:
            return True
        else:
            return False
    
    def _is_cloudice_epoch(self, epoch):
        if epoch >= self.strategy["cloudice_start"] and epoch < self.strategy["cloudice_end"]:
            return True
        else:
            return False
    
    def _is_mix_epoch(self, epoch):
        if epoch >= self.strategy["mix_start"] and epoch < self.strategy["mix_end"]:
            return True
        else:
            return False

    def _is_weight_mix_epoch(self, epoch):
        if epoch >= self.strategy["weighted_mix_start"] and epoch < self.strategy["weighted_mix_end"]:
            return True
        else:
            return False


class MTLLoss_CW(MTLLoss):
    def __init__(self, loss_reg, loss_clsf, strategy, loss_weight):
        super().__init__(loss_reg, loss_clsf, strategy)
        self.loss_reg = loss_reg
        self.loss_clsf  = loss_clsf
        self.strategy = strategy
        self.loss_weight = loss_weight
    
    def forward(self, p_rainrate, p_rainmask, p_cloudwater, l_rainrate, l_rainmask, l_cloudwater, epoch):
        if self._is_rainrate_epoch(epoch):
            batch_loss = self.loss_reg(p_rainrate, l_rainrate)
            loss_type = "rainrate"
        elif self._is_rainmask_epoch(epoch):
            batch_loss = self.loss_clsf(p_rainmask, l_rainmask)
            loss_type = "rainmask"
        elif self._is_cloudwater_epoch(epoch):
            batch_loss = self.loss_reg(p_cloudwater, l_cloudwater)
            loss_type = "cloudwater"
        elif self._is_mix_epoch(epoch):
            loss_rainrate = self.loss_reg(p_rainrate, l_rainrate)
            loss_rainmask = self.loss_clsf(p_rainmask, l_rainmask)
            loss_cloudwater = self.loss_reg(p_cloudwater, l_cloudwater)
            batch_loss = self.loss_weight[0]*loss_rainrate + self.loss_weight[1]*loss_rainmask + self.loss_weight[2]*loss_cloudwater
            loss_type = "mix"
        else:
            raise RuntimeError("undefined epoch num!")
        return batch_loss
    
    def get_lossbreak(self, p_rainrate, p_rainmask, p_cloudwater, l_rainrate, l_rainmask, l_cloudwater, epoch):
        if self._is_mix_epoch(epoch):
            loss_rainrate = self.loss_reg(p_rainrate, l_rainrate)
            loss_rainmask = self.loss_clsf(p_rainmask, l_rainmask)
            loss_cloudwater = self.loss_reg(p_cloudwater, l_cloudwater)
            return [epoch, loss_rainrate.item(), loss_rainmask.item(), loss_cloudwater.item()]
        else:
            return [epoch, np.nan, np.nan, np.nan]


class MTLLoss_CI(MTLLoss):
    def __init__(self, loss_reg, loss_clsf, strategy, loss_weight):
        super().__init__(loss_reg, loss_clsf, strategy)
        self.loss_reg = loss_reg
        self.loss_clsf  = loss_clsf
        self.strategy = strategy
        self.loss_weight = loss_weight
    
    def forward(self, p_rainrate, p_rainmask, p_cloudice, l_rainrate, l_rainmask, l_cloudice, epoch):
        if self._is_rainrate_epoch(epoch):
            batch_loss = self.loss_reg(p_rainrate, l_rainrate)
            loss_type = "rainrate"
        elif self._is_rainmask_epoch(epoch):
            batch_loss = self.loss_clsf(p_rainmask, l_rainmask)
            loss_type = "rainmask"
        elif self._is_cloudice_epoch(epoch):
            batch_loss = self.loss_reg(p_cloudice, l_cloudice)
            loss_type = "cloudice"
        elif self._is_mix_epoch(epoch):
            loss_rainrate = self.loss_reg(p_rainrate, l_rainrate)
            loss_rainmask = self.loss_clsf(p_rainmask, l_rainmask)
            loss_cloudice = self.loss_reg(p_cloudice, l_cloudice)
            batch_loss = self.loss_weight[0]*loss_rainrate + self.loss_weight[1]*loss_rainmask + self.loss_weight[2]*loss_cloudice
            loss_type = "mix"
        else:
            raise RuntimeError("undefined epoch num!")
        return batch_loss
    
    def get_lossbreak(self, p_rainrate, p_rainmask, p_cloudice, l_rainrate, l_rainmask, l_cloudice, epoch):
        if self._is_mix_epoch(epoch):
            loss_rainrate = self.loss_reg(p_rainrate, l_rainrate)
            loss_rainmask = self.loss_clsf(p_rainmask, l_rainmask)
            loss_cloudice = self.loss_reg(p_cloudice, l_cloudice)
            return [epoch, loss_rainrate.item(), loss_rainmask.item(), loss_cloudice.item()]
        else:
            return [epoch, np.nan, np.nan, np.nan]



class MTLLoss_CWCI(MTLLoss):
    def __init__(self, loss_reg, loss_clsf, strategy, loss_weight):
        super().__init__(loss_reg, loss_clsf, strategy)
        self.loss_reg = loss_reg
        self.loss_clsf  = loss_clsf
        self.strategy = strategy
        self.loss_weight = loss_weight
    
    def forward(self, p_rainrate, p_rainmask, p_cloudwater, p_cloudice, l_rainrate, l_rainmask, l_cloudwater, l_cloudice, epoch):
        if self._is_rainrate_epoch(epoch):
            batch_loss = self.loss_reg(p_rainrate, l_rainrate)
            loss_type = "rainrate"
        elif self._is_rainmask_epoch(epoch):
            batch_loss = self.loss_clsf(p_rainmask, l_rainmask)
            loss_type = "rainmask"
        elif self._is_cloudwater_epoch(epoch):
            batch_loss = self.loss_reg(p_cloudwater, l_cloudwater)
            loss_type = "cloudwater"
        elif self._is_cloudice_epoch(epoch):
            batch_loss = self.loss_reg(p_cloudice, l_cloudice)
            loss_type = "cloudice"
        elif self._is_mix_epoch(epoch):
            loss_rainrate = self.loss_reg(p_rainrate, l_rainrate)
            loss_rainmask = self.loss_clsf(p_rainmask, l_rainmask)
            loss_cloudwater = self.loss_reg(p_cloudwater, l_cloudwater)
            loss_cloudice = self.loss_reg(p_cloudice, l_cloudice)
            batch_loss = self.loss_weight[0]*loss_rainrate + self.loss_weight[1]*loss_rainmask + self.loss_weight[2]*loss_cloudwater + self.loss_weight[3]*loss_cloudice
            loss_type = "mix"
        else:
            raise RuntimeError("undefined epoch num!")
        return batch_loss
    
    def get_lossbreak(self, p_rainrate, p_rainmask, p_cloudwater, p_cloudice, l_rainrate, l_rainmask, l_cloudwater, l_cloudice, epoch):
        if self._is_mix_epoch(epoch):
            loss_rainrate = self.loss_reg(p_rainrate, l_rainrate)
            loss_rainmask = self.loss_clsf(p_rainmask, l_rainmask)
            loss_cloudwater = self.loss_reg(p_cloudwater, l_cloudwater)
            loss_cloudice = self.loss_reg(p_cloudice, l_cloudice)
            return [epoch, loss_rainrate.item(), loss_rainmask.item(), loss_cloudwater.item(), loss_cloudice.item()]
        else:
            return [epoch, np.nan, np.nan, np.nan, np.nan]


class MTLLoss_CW_Weighting(MTLLoss):
    def __init__(self, loss_reg, loss_clsf, strategy, loss_weight, alpha, beta):
        super().__init__(loss_reg, loss_clsf, strategy)
        self.loss_reg = loss_reg
        self.loss_clsf = loss_clsf
        self.strategy = strategy
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.beta = beta

    def weighted_loss_reg(self, p_rainrate, p_cloudwater, l_cloudwater):
        squared_error = (p_cloudwater - l_cloudwater)**2
        p_rainrate4weighting = p_rainrate.detach()
        weighting_factor = self.weighting_func(p_rainrate4weighting, self.alpha, self.beta)
        mean_weighted_loss = torch.mean(squared_error * weighting_factor)
        return mean_weighted_loss

    def weighting_func(self, x, alpha, beta):
        """
        alpha = line-smoothing parameter
        beta = threshold for weighting
        """
        y = 1/(1+torch.exp(-(x-beta)*alpha))
        return -1*y + 1

    def forward(self, p_rainrate, p_rainmask, p_cloudwater, l_rainrate, l_rainmask, l_cloudwater, epoch):
        if self._is_rainrate_epoch(epoch):
            batch_loss = self.loss_reg(p_rainrate, l_rainrate)
            loss_type = "rainrate"
        elif self._is_rainmask_epoch(epoch):
            batch_loss = self.loss_clsf(p_rainmask, l_rainmask)
            loss_type = "rainmask"
        elif self._is_cloudwater_epoch(epoch):
            batch_loss = self.loss_reg(p_cloudwater, l_cloudwater)
            loss_type = "cloudwater"
        elif self._is_mix_epoch(epoch):
            loss_rainrate = self.loss_reg(p_rainrate, l_rainrate)
            loss_rainmask = self.loss_clsf(p_rainmask, l_rainmask)
            loss_cloudwater = self.loss_reg(p_cloudwater, l_cloudwater)
            batch_loss = self.loss_weight[0]*loss_rainrate + self.loss_weight[1]*loss_rainmask + self.loss_weight[2]*loss_cloudwater
            loss_type = "mix"
        elif self._is_weight_mix_epoch(epoch):
            loss_rainrate = self.loss_reg(p_rainrate, l_rainrate)
            loss_rainmask = self.loss_clsf(p_rainmask, l_rainmask)
            loss_cloudwater_weigted = self.weighted_loss_reg(p_rainrate, p_cloudwater, l_cloudwater)
            batch_loss = self.loss_weight[0]*loss_rainrate + self.loss_weight[1]*loss_rainmask + self.loss_weight[2]*loss_cloudwater_weigted
            loss_type = "weight_mix"
        else:
            raise RuntimeError("undefined epoch num!")
        return batch_loss

    def get_lossbreak(self, p_rainrate, p_rainmask, p_cloudwater, l_rainrate, l_rainmask, l_cloudwater, epoch):
        if self._is_mix_epoch(epoch):
            loss_rainrate = self.loss_reg(p_rainrate, l_rainrate)
            loss_rainmask = self.loss_clsf(p_rainmask, l_rainmask)
            loss_cloudwater = self.loss_reg(p_cloudwater, l_cloudwater)
            return [epoch, loss_rainrate.item(), loss_rainmask.item(), loss_cloudwater.item()]
        elif self._is_weight_mix_epoch(epoch):
            loss_rainrate = self.loss_reg(p_rainrate, l_rainrate)
            loss_rainmask = self.loss_clsf(p_rainmask, l_rainmask)
            loss_cloudwater_weigted = self.weighted_loss_reg(p_rainrate, p_cloudwater, l_cloudwater)
            return [epoch, loss_rainrate.item(), loss_rainmask.item(), loss_cloudwater_weigted.item()]
        else:
            return [epoch, np.nan, np.nan, np.nan]

