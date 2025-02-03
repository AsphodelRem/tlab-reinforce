import torch
import hydra
import comet_ml
from omegaconf import DictConfig

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_quantization_config,
    setup_chat_format,
)

from src.dataset import prepare_dataset
from src.trainers.reward_model_pairwised import PairwisedRewardTrainer
from src.utils.init_utils import set_random_seed, set_worker_seed


@hydra.main(version_base=None, config_path="configs", config_name="reward_model")
def main(config: DictConfig):
    set_random_seed(42)
    set_worker_seed(0)
    comet_ml.login(project_name="tlab-reinforce-reward-model")
    
    training_args = RewardConfig(**config.trainer.training_args)
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    tokenizer = AutoTokenizer.from_pretrained(
        config.reward_model.model_name,
        **config.reward_model.model_params
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model.model_name,
        **config.reward_model.model_params,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    _, _, train_ds, test_ds = prepare_dataset(**config.dataset)

    TrainerClass = (
        PairwisedRewardTrainer
        if config.reward_model.model_params.num_labels != 1
        else RewardTrainer
    )
    
    trainer = TrainerClass(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )

    trainer.train()

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=config.dataset.dataset_name)

if __name__ == "__main__":
    main()