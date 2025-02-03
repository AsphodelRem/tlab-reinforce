import comet_ml
import hydra
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorWithPadding
)
from trl import ModelConfig, get_peft_config
from peft import LoraConfig, get_peft_model 

from src.dataset import prepare_dataset
from src.trainers.reinforce_trainer import ReinforceTrainer
from src.utils.init_utils import set_random_seed, set_worker_seed
from src import loss


@hydra.main(version_base=None, config_path="configs", config_name="alignment")
def main(config):
    set_random_seed(42)
    set_worker_seed(0)
    comet_ml.login(project_name="tlab-reinforce-align")

    sft = AutoModelForCausalLM.from_pretrained(
        config.sft.model_name, 
        **config.sft.model_params, 
    ).to('cuda:1')
    tokenizer = AutoTokenizer.from_pretrained(
        config.sft.model_name, 
        **config.sft.tokenizer
    )
    
    peft_config = LoraConfig(
        **config.trainer.model_config
    )
    sft = get_peft_model(sft, peft_config)
    sft.print_trainable_parameters() 

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model.model_name, 
        torch_dtype=torch.bfloat16
    ).to('cuda:1')
    reward_model_tokenizer = AutoTokenizer.from_pretrained(
        config.reward_model.model_name, 
        **config.reward_model.tokenizer
    )

    train_ds, test_ds, _, _ = prepare_dataset(**config.dataset)

    def tokenize_function(examples):
        return tokenizer(
            examples["prompt"], 
            truncation=True, 
            padding=True, 
            max_length=512
        )

    train_ds = train_ds.map(
        tokenize_function, 
        batched=True
    ).remove_columns(['prompt'])
    
    test_ds = test_ds.map(
        tokenize_function, 
        batched=True
    ).remove_columns(['prompt'])
    
    test_ds = test_ds.select(range(128))

    training_args = TrainingArguments(**config.trainer.training_args)

    trainer = ReinforceTrainer(
        model=sft,
        args=training_args,
        tokenizer=tokenizer,
        reward_model=reward_model,
        reward_model_tokenizer=reward_model_tokenizer,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        loss_class='reinforce_loss',
    )
    
    print("Start evaluation to get SFT mean reward")
    print(f"SFT metrics: {trainer.evaluate()}")
    
    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()
