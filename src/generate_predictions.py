import json
from datasets import load_dataset
from src.logger import get_logger

logger = get_logger(__name__)

def load_test_set(config: dict):
    """Loads the raw test set (not tokenized — we need raw input/output for eval)."""
    eval_cfg = config["evaluation"]
    logger.info(f"Loading test set from {eval_cfg['test_path']}")
    test_dataset = load_dataset("json", data_files=eval_cfg["test_path"], split="train")
    logger.info(f"Test examples: {len(test_dataset)}")
    return test_dataset


def generate_prediction(example: dict, tokenizer, model, config: dict) -> str:
    """Generates a JSON prediction for one test input, deterministically (no sampling)."""
    eval_cfg = config["evaluation"]

    system_message = (
        "You are a robot command parser. Convert the user's natural language "
        "command into structured JSON representing the robot action(s)."
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": example["input"]},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # add_generation_prompt=True -> tells the template to leave the "assistant turn"
    # open, so the model knows it's its turn to generate, not repeat the prompt

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=eval_cfg["max_new_tokens"],
        do_sample=False,     # greedy decoding — deterministic, no randomness
        temperature=None,    # explicitly unused when do_sample=False
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],   # sirf naya generated part, prompt wapas nahi
        skip_special_tokens=True,
    )
    return generated_text.strip()


def generate_all_predictions(test_dataset, tokenizer, model, config: dict):
    """Runs generation over the full test set, returns list of {input, expected, predicted}."""
    logger.info("Generating predictions on test set")
    results = []

    for i, example in enumerate(test_dataset):
        prediction = generate_prediction(example, tokenizer, model, config)
        results.append({
            "input": example["input"],
            "expected": example["output"],
            "predicted_raw": prediction,
        })
        if (i + 1) % 10 == 0:
            logger.info(f"Generated {i + 1}/{len(test_dataset)} predictions")

    return results