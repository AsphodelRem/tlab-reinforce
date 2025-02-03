import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
)
from tqdm import tqdm
from typing import Dict, List, Union, Optional

from src import loss


class ReinforceTrainer(Trainer):
    """
    Trainer class for align LLM via REINFORCE algorithm using a reward model.
    """
    
    def __init__(
        self,
        model: AutoModel,
        args: TrainingArguments,
        tokenizer: AutoTokenizer,
        reward_model: AutoModelForSequenceClassification,
        reward_model_tokenizer: AutoTokenizer,
        train_dataset = None,
        eval_dataset = None,
        data_collator = None,
        loss_class: str = None,
        **kwargs,
    ):
        """
        Initializes the ReinforceTrainer.

        Args:
            model (AutoModel): The main model for text generation.
            args (TrainingArguments): Training configuration and hyperparameters.
            tokenizer (AutoTokenizer): Tokenizer for the main model.
            reward_model (AutoModelForSequenceClassification): Model used for reward evaluation.
            reward_model_tokenizer (AutoTokenizer): Tokenizer for the reward model.
            train_dataset (optional): Dataset for training.
            eval_dataset (optional): Dataset for evaluation.
            data_collator (optional): Function to collate data samples into batches.
            loss_class (str, optional): Loss function to be used.
            **kwargs: Additional arguments passed to the Trainer superclass.
        """
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            **kwargs,
        )
        
        self.reward_model = reward_model.to(self.args.device)
        self.reward_model_tokenizer = reward_model_tokenizer
        self.loss = loss.LossRegister.get_registered_by_name(loss_class)()
        self.saved_reward: List[float] = []

        self.reward_model.eval()
        for param in self.reward_model.parameters():
            param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Computes the loss based on generated completions and reward values.

        Args:
            model: The main language model.
            inputs: Input data for generation.
            return_outputs (bool, optional): If True, returns the model outputs along with loss.
            **kwargs: Additional arguments.

        Returns:
            Loss value or a tuple of loss and model outputs.
        """
        completions, logps = self._generate_completion(inputs)
        rewards = self._compute_reward(completions)
        baseline = self._compute_baseline()
        
        advantage = rewards - baseline
        
        loss_value = self.loss(logps, advantage)
        self.saved_reward.append(rewards)
        
        return (loss_value, None) if return_outputs else loss_value

    def _compute_baseline(self) -> float:
        """
        Computes the baseline reward as the average of previously stored rewards.

        Returns:
            float: The computed baseline value.
        """
        return sum(self.saved_reward) / len(self.saved_reward) if len(self.saved_reward) > 0 else 0.0

    def _generate_completion(self, inputs: List[str]) -> tuple[List[str], torch.Tensor]:
        """
        Generates text completions from the model and computes log probabilities.

        Args:
            inputs (List[str]): List of input strings for generation.

        Returns:
            Tuple[List[str], torch.Tensor]: Generated completions and their log probabilities.
        """
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=256,
            pad_token_id=self.processing_class.eos_token_id,
        )

        log_probs = self._calculate_log_probs(generated_ids)
        
        completions = self.processing_class.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )

        return completions, log_probs

    def _calculate_log_probs(self, generated_ids: torch.Tensor) -> torch.Tensor:
        """
        Computes log probabilities for generated tokens.

        Args:
            generated_ids (torch.Tensor): Tensor of generated token IDs.

        Returns:
            torch.Tensor: Log probabilities of the generated tokens.
        """
        logits = self.model(generated_ids).logits
        return torch.nn.functional.log_softmax(logits, dim=-1)

    def _compute_reward(self, completions: List[str]) -> torch.Tensor:
        """
        Computes rewards using the reward model.

        Args:
            completions (List[str]): List of generated text completions.

        Returns:
            torch.Tensor: Computed reward values.
        """
        inputs = self.reward_model_tokenizer(
            completions,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.args.device)

        with torch.no_grad():
            outputs = self.reward_model(**inputs)
            rewards = torch.sigmoid(outputs.logits).mean()

        return rewards

    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        """
        Custom evaluation function that tracks reward values.

        Returns:
            Dict[str, float]: Dictionary containing mean reward and evaluation loss.
        """
        dataloader = self.get_eval_dataloader(self.eval_dataset)
        eval_rewards, eval_losses = [], []
        for i, batch in enumerate(tqdm(dataloader), 1):
            with torch.no_grad():
                completions, logps = self._generate_completion(batch)
                reward = self._compute_reward(completions)
                loss = self.loss(logps, reward)
                eval_rewards.append(reward.mean().item())
                eval_losses.append(loss.item())

        metrics = {
            "mean_reward": torch.tensor(eval_rewards).mean().item(), 
            "eval_loss": torch.tensor(eval_losses).mean().item()
        }

        self.log(metrics)
        return metrics
