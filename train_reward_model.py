import torch
import hydra

from transformers import  (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    HfArgumentParser
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


@hydra.main(version_base=None, config_path="configs", config_name="reward_modeling")
def main(config):
    parser = HfArgumentParser((ScriptArguments, RewardConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=model_args.trust_remote_code, 
        use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, 
        num_labels=1, 
        trust_remote_code=model_args.trust_remote_code, 
        **model_kwargs
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    _, _, train_ds, test_ds = prepare_dataset(**config.dataset)

 
    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds
    )
    trainer.train()

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()

