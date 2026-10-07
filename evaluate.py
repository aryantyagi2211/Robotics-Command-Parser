from src.config_loader import load_config
from src.inference import load_trained_model
from src.logger import get_logger
from src.generate_predictions import load_test_set, generate_all_predictions
from src.parse_and_compare import parse_all_predictions, compare_all
from src.save_eval_report import save_eval_report


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

field_names = ["action", "object", "target", "location", "quantity", "attributes"]
field_correct = {f: 0 for f in field_names}
field_total = {f: 0 for f in field_names}

for r in results:
    for i, is_correct in enumerate(r["field_results"]):
        field = field_names[i % len(field_names)]
        field_total[field] += 1
        if is_correct:
            field_correct[field] += 1

logger.info("Field-level accuracy:")
for f in field_names:
    if field_total[f] > 0:
        pct = 100 * field_correct[f] / field_total[f]
        logger.info(f"  {f}: {field_correct[f]}/{field_total[f]} ({pct:.1f}%)")

# simple vs complex breakdown
simple_results = [r for r in results if "actions" not in r["expected"]]
complex_results = [r for r in results if "actions" in r["expected"]]

simple_acc = 100 * sum(r["exact_match"] for r in simple_results) / len(simple_results) if simple_results else 0
complex_acc = 100 * sum(r["exact_match"] for r in complex_results) / len(complex_results) if complex_results else 0

logger.info(f"Simple/medium commands: {sum(r['exact_match'] for r in simple_results)}/{len(simple_results)} ({simple_acc:.1f}%)")
logger.info(f"Complex (multi-action) commands: {sum(r['exact_match'] for r in complex_results)}/{len(complex_results)} ({complex_acc:.1f}%)")

metrics = {
    "total": len(results),
    "valid_json": sum(r["is_valid_json"] for r in results),
    "valid_json_pct": 100 * sum(r["is_valid_json"] for r in results) / len(results),
    "exact_match": exact_matches,
    "exact_match_pct": 100 * exact_matches / len(results),
    "field_accuracy": {f: (field_correct[f], field_total[f]) for f in field_names},
    "simple_correct": sum(r["exact_match"] for r in simple_results),
    "simple_total": len(simple_results),
    "simple_pct": simple_acc,
    "complex_correct": sum(r["exact_match"] for r in complex_results),
    "complex_total": len(complex_results),
    "complex_pct": complex_acc,
}

save_eval_report(results, metrics)