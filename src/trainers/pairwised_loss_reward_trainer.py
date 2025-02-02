from typing import Any, Union

import torch
from transformers import PreTrainedModel
from trl import RewardTrainer


class PairwisedRewardTrainer(RewardTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, Union[torch.Tensor, Any]],
        **kwargs,
    ) -> torch.Tensor:
        p_win = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
        )["logits"]
        p_loss = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
            return_dict=True,
        )["logits"]

        joint = p_win.unsqueeze(2) * p_loss.unsqueeze(1)

        # Создаем маску, где для позиций (i, j) стоит 1, если i > j, и 0 иначе.
        num_ratings = p_win.size(1)
        ratings = torch.arange(num_ratings, device=p_win.device)
        mask = (ratings.unsqueeze(1) > ratings.unsqueeze(0)).float()

        # Применяем маску: суммируем по i и j
        pairwise_prob = (joint * mask).sum(dim=(1, 2))

        loss = -torch.log(pairwise_prob + 1.0e-8).mean()
        return loss
