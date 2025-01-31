import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    AutoModelForSequenceClassification,
)
from tqdm import tqdm
from typing import Union

import loss


class ReinforceTrainer(Trainer):
    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        reward_model: AutoModelForSequenceClassification,
        reward_model_tokenizer: AutoTokenizer,
        train_dataset,
        eval_dataset,
        training_args: dict,
        **kwargs,
    ):
        self.sft = model
        self.sft_tokenizer = tokenizer
        self.reward_model = reward_model
        self.reward_model_tokenizer = reward_model_tokenizer

        self.sft.to("cuda:0")
        self.reward_model.to("cuda:0")

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_args = training_args

        self.saved_reward = []

        self.loss = loss.LossRegister.get_registered_by_name(training_args["loss"])()
        self.optimizer = self._create_optimizer()

    def train(self):
        train_dataloader = self._get_dataloader(self.train_dataset)
        self.sft.train()
        for batch in tqdm(train_dataloader):
            self.sft.zero_grad()
            loss, reward = self.proccess_batch(batch)
            self.saved_reward.append(reward.mean())
            loss.backward()
            self.optimizer.step()

            del batch, reward
            torch.cuda.empty_cache()

    def evaluate(self):
        eval_dataloader = self._get_dataloader(self.eval_dataset)
        for batch in tqdm(eval_dataloader):
            eval_reward, eval_loss = [], []
            with torch.no_grad():
                loss, reward = self.proccess_batch(batch)
                eval_reward.append(reward.mean())
                eval_loss.append(loss)
            del batch, reward
            torch.cuda.synchronize()

        print(
            f"Eval mean reward: {torch.tensor(eval_reward).mean()}, \
            eval loss: {torch.tensor(eval_loss).mean()}"
        )

    def proccess_batch(self, batch: dict) -> Union[torch.tensor, torch.tensor]:
        batch = batch["prompt"]
        outputs, logps = self._generate_completion(batch)

        reward = self._compute_reward(outputs)
        baseline = self._compute_baseline()
        loss = self.loss(logps, reward, baseline)
        del outputs, logps

        return loss, reward

    def _get_dataloader(
        self, 
        dataset: list, 
        data_collator: callable = None
    ) -> DataLoader:
        """
        Create DataLoader for the given dataset.

        Args:
        dataset (torch.utils.data.Dataset): The dataset to load.
        data_collator (optional): Function to collate data.

        Returns:
        torch.utils.data.DataLoader: The DataLoader for the dataset.
        """
        dataloader = DataLoader(
            dataset,
            batch_size=self.training_args.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            drop_last=True,
        )
        return dataloader

    def _create_optimizer(self):
        decay_parameters = self.get_decay_parameter_names(self.sft)
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.sft.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": self.training_args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.sft.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        return Adam(optimizer_grouped_parameters, lr=self.training_args.learning_rate)

    def _compute_baseline(self):
        return (
            sum(self.saved_reward) / len(self.saved_reward)
            if len(self.saved_reward) > 0
            else 0
        )

    def _generate_completion(
        self, 
        batched_prompts: list[str]
    ) -> tuple[list[str], torch.Tensor]:
        """
        Generate text completions and computes log probabilities.

        Args:
        - batched_prompts (list[str]): Batch of initial text prompts.

        Returns:
        - list[str]: Generated text completions.
        - torch.Tensor: Log probabilities of the generated words.
        """

        model_input = self.sft_tokenizer(
            batched_prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(self.device)

        outputs = self.sft.generate(
            **model_input,
            pad_token_id=self.sft_tokenizer.eos_token_id,
            max_new_tokens=256,
        )
        decoded_completions = self.sft_tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        log_probs = self._get_batch_logps(outputs)

        return decoded_completions, log_probs

    def _get_batch_logps(
            self, 
            batched_completions: torch.Tensor
        ) -> torch.Tensor:
        """
        Compute log probabilities of words for a batch of completions.

        Args:
        batched_completions (torch.Tensor): Batch of generated completions.

        Returns:
        torch.Tensor: Log probabilities of the words.
        """
        logits = self.sft(batched_completions).logits
        return torch.nn.functional.log_softmax(logits, dim=-1)

    def _compute_reward(self, batch: list[str]) -> torch.Tensor:
        with torch.no_grad():
            model_input = self.reward_model_tokenizer(
                batch, 
                truncation=True, 
                padding=True, 
                return_tensors="pt"
            ).to("cuda:0")

            return torch.sigmoid(self.reward_model(**model_input).logits)
