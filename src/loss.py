from typing import Dict, Type, Tuple

import torch
import torch.nn as nn


class LossRegister:
    _registry: Dict[str, Tuple[Type, str | None]] = {}
    
    @classmethod
    def register(cls, name):
        def decorator(subclass: Type):
            if name in cls._registry:
                raise ValueError(f"Class with name '{name}' is already registered.")
            cls._registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def get_registered_by_name(self, name: str):
        return self._registry.get(name)


@LossRegister.register('reinforce_loss')
class ReinforceLoss(nn.Module):
    def __init__(self):
        pass

    def __call__(self, log_probs, reward, baseline):  
        log_probs_total = log_probs.sum(dim=1)
        return -(log_probs_total * (reward - baseline)).mean()


@LossRegister.register('dpo_loss')
class DPOLoss(nn.Module):
    def __init__(self, beta: float=1.0):
        self.beta = beta

    def __call__(self, reward_l, gt_l, reward_w, gt_w):
        return -(
            self.beta * torch.log(reward_l / gt_l) - self.beta * torch.log(reward_w / gt_w)
        ).mean()
        