from src.config_loader import load_config
from src.model_setup import load_base_model, attach_lora_adapters
from src.data_loader import load_raw_dataset
from src.data_formatter import format_dataset
from src.tokenize_data import tokenize_dataset
from src.training_args_builder import build_training_args
from src.trainer_setup import build_trainer, run_training
from src.logger import get_logger

logger = get_logger(__name__)

# --- Setup ---
config = load_config("config/config.yaml")
tokenizer, base_model = load_base_model(config)
model = attach_lora_adapters(base_model, config)

# --- Data ---
train_dataset, val_dataset = load_raw_dataset(config)
train_dataset = format_dataset(train_dataset, tokenizer)
val_dataset = format_dataset(val_dataset, tokenizer)
train_dataset = tokenize_dataset(train_dataset, tokenizer, config)
val_dataset = tokenize_dataset(val_dataset, tokenizer, config)

# --- Training ---
training_args = build_training_args(config)
trainer = build_trainer(model, tokenizer, train_dataset, val_dataset, training_args)
run_training(trainer)

logger.info("Pipeline complete — model trained and checkpoints saved")