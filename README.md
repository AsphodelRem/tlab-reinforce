# Implementation of REINFORCE with baseline for LLM alignment

[Original paper](https://arxiv.org/abs/2402.14740)

## Structure:

### `configs/`
- **alignment_with_custom_rm.yaml**: Configuration file for alignment with a custom reward model.
- **alignment.yaml**: General alignment configuration file.
- **custom_reward_model.yaml**: Configuration file for a custom reward model.
- **reward_model.yaml**: Configuration file for the reward model.

### `report/`
- **report.md**: Markdown file for reporting project details, results, and analysis.
- 
### `src/`
- **dataset.py**: Script for managing and processing datasets.
- **loss.py**: Script for defining and calculating loss functions.
- 
### `src/trainers/`
- **reinforce_trainer.py**: Python script for training using the reinforcement learning approach.
- **reward_model_pairwised.py**: Python script for training the reward model using pairwise comparisons.

### `src/utils/`
- **init_utils.py**: Initialization utilities for the project.
- **io_utils.py**: Input/Output utilities for handling data.

- **run_reinforce.py**: Script to run the reinforcement learning training.
- **train_reward_model.py**: Script to train the reward model.



## How to reproduce:
```
git clone git@github.com:AsphodelRem/tlab-reinforce.git
cd tlab-reinforce

python3 -m venv .venv
. venv/bin/activate
pip3 install -r requirements.txt

# Train reward model (may requires lots of GPU memory with default config)
python3 train_reward_model.py

# Run alignment with saved reward model (you can specify path to trained reward model in in corresponding config file, default one leads to huggingface model)
python3 run_reinforce.py 

# Train custom reward model
python3 train_reward_model.py --config-name custom_reward_model.yaml

# Run alignment with saved custom reward model (you can specify path to trained reward model in corresponding config file, default one leads to huggingface model)
python3 run_reinforce.py --config-name alignment_with_custom_rm.yaml
```
