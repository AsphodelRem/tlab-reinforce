import hydra
import torch
import comet_ml
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification
)

from src.dataset import prepare_dataset
from src.trainers.reinforce_trainer import ReinforceTrainer
from src.utils.init_utils import set_random_seed, set_worker_seed


@hydra.main(version_base=None, config_path="configs", config_name="alignment")
def main(config):
    set_random_seed(42)
    set_worker_seed(0)
    comet_ml.login(project_name="tlab-reinforce-align")
    
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
        **config.trainer
    )

    trainer.evaluate()
    trainer.train()
    trainer.evaluate()


if __name__ == '__main__':
    main()