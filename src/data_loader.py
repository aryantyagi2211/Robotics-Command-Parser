from datasets import load_dataset
from src.logger import get_logger

logger = get_logger(__name__)

def load_raw_dataset(config: dict):
    """load train and val JSONL files into huggingface dataset objects"""
    data_cfg = config["data"]

    logger.info(f"Loading train data from {data_cfg['train_path']}")
    train_dataset = load_dataset("json", data_files=data_cfg["train_path"], split="train")

    logger.info(f"Loading val data from {data_cfg['val_path']}")
    val_dataset = load_dataset("json", data_files=data_cfg["val_path"], split="train")

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    return train_dataset, val_dataset