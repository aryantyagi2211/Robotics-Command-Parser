from src.logger import get_logger

logger = get_logger(__name__)

def tokenize_dataset(dataset, tokenizer, config: dict):
    data_cfg = config["data"]
    max_length = data_cfg["max_length"]

    def tokenize_fn(example):
        tokenized = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return tokenized

    logger.info(f"Tokenizing dataset with max_length={max_length}")
    train_dataset = dataset.map(
        tokenize_fn,
        remove_columns=dataset.column_names,
    )
    return train_dataset
