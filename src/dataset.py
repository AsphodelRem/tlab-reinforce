from datasets import load_dataset


def prepare_dataset(
    dataset_name: str,  
    split: str, 
    test_size=0.2
):
    """
    Creates train and eval partition
    """
    
    dataset = load_dataset(dataset_name, split=split)
    
    def add_margin(row):
        return {'margin': row['chosen_rating'] - row['rejected_rating']}

    dataset = dataset.map(add_margin).train_test_split(test_size, seed=42)

    prompt_ds_train, prompt_ds_test = (
        dataset["train"].select_columns(["prompt"]),
        dataset["test"].select_columns(["prompt"]),
    )
    pair_ds_train, pair_ds_test = (
        dataset["train"].select_columns(["chosen", "rejected", "margin"]),
        dataset["test"].select_columns(["chosen", "rejected", "margin"]),
    )
    return prompt_ds_train, prompt_ds_test, pair_ds_train, pair_ds_test
    # return {
    #     'prompts_train': prompt_ds_train, 
    #     'prompts_test': prompt_ds_test, 
    #     'pair_train': pair_ds_train, 
    #     'pair_test': pair_ds_test
    # }
    