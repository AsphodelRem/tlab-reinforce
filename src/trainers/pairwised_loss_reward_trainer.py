from typing import Any, Union

import torch
from tqdm import tqdm
from transformers import PreTrainedModel
from trl import RewardTrainer


class PairwisedRewardTrainer(RewardTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_pairwise_prob(
        self,
        logits_chosen,
        logits_rejected,
        return_input_probs=False
    ):
        p_win = torch.softmax(logits_chosen, dim=-1)
        p_loss = torch.softmax(logits_rejected, dim=-1)
        joint = p_win.unsqueeze(2) * p_loss.unsqueeze(1)

        # Создаем маску, где для позиций (i, j) стоит 1, если i > j, и 0 иначе.
        num_ratings = p_win.size(1)
        ratings = torch.arange(num_ratings, device=p_win.device)
        mask = (ratings.unsqueeze(1) > ratings.unsqueeze(0)).float()

        # Применяем маску: суммируем по i и j
        pairwise_prob = (joint * mask).sum(dim=(1, 2))

        if return_input_probs:
            return pairwise_prob, p_win, p_loss
        return pairwise_prob

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        **kwargs,
    ) -> torch.Tensor:
        p_win = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
            return_dict=True,
        ).logits
        p_loss = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
            return_dict=True,
        ).logits

        pairwise_probs = self.get_pairwise_prob(p_win, p_loss)
        loss = -torch.log(pairwise_probs + 1.0e-8).mean()

        if return_outputs:
            return loss, {
                "rewards_chosen": p_win,
                "rewards_rejected": p_loss,
            }
        return loss

    def evaluate(self, **kwargs) -> dict:
        """
        Custom evaluation method for the reward trainer.

        This method computes:
          - Average Negative Log-Likelihood (NLL) over the evaluation dataset.
          - Pairwise Accuracy: the percentage of pairs where the expected rating
            (i.e., expectation under the predicted distribution) for the chosen text
            is higher than for the rejected text.

        Returns:
            dict: A dictionary of evaluation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_pairs = 0
        correct_pairs = 0
        eps = 1e-8

        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluation"):
                logits_chosen = self.model(
                    input_ids=batch["input_ids_chosen"],
                    attention_mask=batch["attention_mask_chosen"],
                    return_dict=True,
                ).logits
                logits_rejected = self.model(
                    input_ids=batch["input_ids_rejected"],
                    attention_mask=batch["attention_mask_rejected"],
                    return_dict=True,
                ).logits

                pairwise_probs, p_chosen, p_rejected = self.get_pairwise_prob(
                    logits_chosen,
                    logits_rejected,
                    return_input_probs=True
                )
                loss = -torch.log(pairwise_probs + eps)

                total_loss += loss.sum().item()
                batch_size = p_chosen.size(0)
                total_pairs += batch_size

                num_ratings = p_chosen.size(1)
                rating_values = torch.arange(
                    1, num_ratings + 1, device=p_chosen.device
                ).float()
                expected_chosen = torch.sum(p_chosen * rating_values, dim=-1)
                expected_rejected = torch.sum(p_rejected * rating_values, dim=-1)

                correct = (expected_chosen > expected_rejected).float()
                correct_pairs += correct.sum().item()

        avg_loss = total_loss / total_pairs
        pairwise_accuracy = correct_pairs / total_pairs

        metrics = {
            "eval_loss": avg_loss,
            "pairwise_accuracy": pairwise_accuracy,
        }

        self.log(metrics)
        return metrics
