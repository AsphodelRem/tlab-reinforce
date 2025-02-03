from typing import ClassVar, Dict, Type, Tuple

import torch
import torch.nn as nn


class LossRegister:
    _registry: Dict[str, Tuple[Type, str | None]] = {}
    
    @classmethod
    def register(cls, name):
        def decorator(subclass: Type):
            if name in cls._registry and not exist_ok:
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

    def __call__(self, log_probs: torch.tensor, reward: torch.tensor):  
        return -(log_probs * reward).mean()

@LossRegister.register('prob_reward_loss')
class CustomRewardLoss(nn.Module):
    def __init__(self):
        pass

    def __call__(self, log_probs: torch.tensor, reward_distribution: torch.tensor):
        num_ratings = reward_distribution.size(0)
        ratings = torch.arange(1, num_ratings + 1, device=reward_distribution.device).float()
        scalar_reward = torch.mean(reward_distribution * ratings, dim=-1)

        return -(log_probs * scalar_reward).mean()




    