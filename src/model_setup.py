import yaml
from src.logger import get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logger = get_logger(__name__)


def load_base_model(config: dict):
    """Loads the tokenizer and base model, quantized to 4-bit per config."""
    model_cfg = config["model"]

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=model_cfg["load_in_4bit"],
        bnb_4bit_quant_type=model_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, model_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=model_cfg["bnb_4bit_use_double_quant"],
    )

    logger.info(f"Loading tokenizer: {model_cfg['base_model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading base model in 4-bit: {model_cfg['base_model_name']}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model_name"],
        quantization_config=quantization_config,
        device_map=model_cfg["device_map"],
    )
    logger.info("Base model loaded successfully")

    return tokenizer, base_model

def attach_lora_adapters(base_model, config: dict):
    """Prepares the 4-bit model for training and attaches LoRA adapters."""
    lora_cfg = config["lora"]

    logger.info("Preparing quantized model for k-bit training")
    base_model = prepare_model_for_kbit_training(base_model)

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
    )

    logger.info(f"Attaching LoRA adapters with r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}")
    model = get_peft_model(base_model, peft_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")

    return model