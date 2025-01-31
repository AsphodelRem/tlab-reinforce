import hydra
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from src.dataset import prepare_dataset
from src.reinforce_trainer import ReinforceTrainer


@hydra.main(version_base=None, config_path="configs", config_name="alignment")
def main(config):
    sft = AutoModelForCausalLM.from_pretrained(
        config.sft.model_name, 
        **config.sft.model_params, 
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.sft.model_name, 
        **config.sft.tokenizer
    )

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model.model_name, 
        torch_dtype=torch.bfloat16
    )
    reward_model_tokenizer = AutoTokenizer.from_pretrained(
        config.reward_model.model_name, 
        **config.reward_model.tokenizer
    )

    train_ds, test_ds, _, _ = prepare_dataset(**config.dataset)

    trainer = ReinforceTrainer(
        sft,
        tokenizer,
        reward_model,
        reward_model_tokenizer,
        train_ds,
        test_ds,
        **config.trainer,
    )

    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()
