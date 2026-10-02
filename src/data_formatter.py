import json
from src.logger import get_logger

logger = get_logger(__name__)

def format_example(example: dict, tokenizer) -> dict:
    """Converts one {input, output} example into a single training text string,
    using Llama's chat template (system + user + assistant turns)."""

    system_message = (
        "You are a robot command parser. Convert the user's natural language "
        "command into structured JSON representing the robot action(s)."
    )
    user_message = example["input"]
    assistant_message = json.dumps(example["output"])  # dict -> JSON string

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message},
    ]

    full_text = tokenizer.apply_chat_template(messages, tokenize=False)

    return {"text": full_text}


def format_dataset(dataset, tokenizer):
    """Applies format_example to every row in a HuggingFace Dataset."""
    logger.info("Formatting dataset into chat-style training text")
    formatted = dataset.map(lambda ex: format_example(ex, tokenizer))
    return formatted