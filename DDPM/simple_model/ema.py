'''
Author: Suizhi HUANG && sunrisen.huang@gmail.com
Date: 2024-03-22 17:49:00
LastEditors: Suizhi HUANG && sunrisen.huang@gmail.com
LastEditTime: 2024-03-22 17:49:00
FilePath: /DL_Demo/DDPM/simple_model/ema.py
Description: exponential moving average (EMA) for model parameters
Copyright (c) 2024 by $Suizhi HUANG, All Rights Reserved. 
'''


class EMA(object):
    def __init__(self, mu=0.99):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.shadow[name].data = (
                    self.mu * self.shadow[name].data + (1 - self.mu) * param.data
                )

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
