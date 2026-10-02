import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.logger import get_logger

logger = get_logger(__name__)

def load_trained_model(config: dict, checkpoint_path: str):
    """Loads the base model (4-bit) and attaches the trained LoRA adapter from checkpoint."""
    model_cfg = config["model"]

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=model_cfg["load_in_4bit"],
        bnb_4bit_quant_type=model_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, model_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=model_cfg["bnb_4bit_use_double_quant"],
    )

    logger.info(f"Loading base model: {model_cfg['base_model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model_name"],
        quantization_config=quantization_config,
        device_map=model_cfg["device_map"],
    )

    logger.info(f"Attaching trained LoRA adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()  # inference mode — dropout off, etc.

    logger.info("Trained model ready for evaluation")
    return tokenizer, model
