from src.config_loader import load_config
from src.model_setup import load_base_model, attach_lora_adapters


config = load_config("config/config.yaml")
tokenizer, base_model = load_base_model(config)
model = attach_lora_adapters(base_model, config)