from transformers import TrainingArguments
from src.logger import get_logger

logger = get_logger(__name__)

def build_training_args(config: dict) -> TrainingArguments:
    """Build HuggingFace training arguments from config"""
    train_cfg = config["training"]

    logger.info(f"Building training arguments")
    args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        eval_strategy=train_cfg["eval_strategy"],
        save_total_limit=train_cfg["save_total_limit"],
        bf16=train_cfg["bf16"],
        fp16=train_cfg["fp16"],
        report_to=train_cfg["report_to"],
    )
    return args