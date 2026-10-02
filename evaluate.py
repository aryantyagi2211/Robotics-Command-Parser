from src.config_loader import load_config
from src.inference import load_trained_model
from src.logger import get_logger
from src.generate_predictions import load_test_set, generate_all_predictions
from src.parse_and_compare import parse_all_predictions, compare_all


logger = get_logger(__name__)

config = load_config("config/config.yaml")

checkpoint_path = config["evaluation"]["checkpoint_path"]
tokenizer, model = load_trained_model(config, checkpoint_path)

logger.info("Checkpoint loaded successfully — ready for evaluation")

test_dataset = load_test_set(config)
results = generate_all_predictions(test_dataset, tokenizer, model, config)

logger.info(f"Generated predictions for {len(results)} test examples")

results = parse_all_predictions(results)

# ek sample dekhte hain — pehla example
logger.info(f"Sample input: {results[0]['input']}")
logger.info(f"Sample expected: {results[0]['expected']}")
logger.info(f"Sample predicted (raw): {results[0]['predicted_raw']}")
logger.info(f"Sample valid JSON: {results[0]['is_valid_json']}")

results = compare_all(results)

exact_matches = sum(1 for r in results if r["exact_match"])
logger.info(f"Exact match: {exact_matches}/{len(results)} ({100*exact_matches/len(results):.1f}%)")