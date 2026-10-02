import json
from src.logger import get_logger

logger = get_logger(__name__)

def extract_json_object(text: str) -> str:
    """Extracts the first complete, balanced JSON object from text,
    ignoring any trailing garbage (extra braces, stray tokens, etc.)."""
    start = text.find("{")
    if start == -1:
        return text  # no opening brace at all, let json.loads fail naturally

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]  # balanced — stop here, ignore rest

    return text[start:]  # never balanced, return what we have


def try_parse_json(text: str):
    """Attempts to parse model output as JSON, after extracting the balanced object."""
    cleaned = extract_json_object(text)
    try:
        parsed = json.loads(cleaned)
        return parsed, True
    except json.JSONDecodeError:
        return None, False


def parse_all_predictions(results: list):
    """Adds parsed JSON + validity flag to each result."""
    logger.info("Parsing model outputs as JSON")
    valid_count = 0

    for r in results:
        parsed, is_valid = try_parse_json(r["predicted_raw"])
        r["predicted_parsed"] = parsed
        r["is_valid_json"] = is_valid
        if is_valid:
            valid_count += 1

    logger.info(f"Valid JSON outputs: {valid_count}/{len(results)}")
    return results

def compare_single(expected: dict, predicted: dict) -> dict:
    """Compares one prediction to its ground truth. Handles both flat and multi-action formats."""

    # normalize both to a list of actions, so single and multi-action compare the same way
    expected_actions = expected["actions"] if "actions" in expected else [expected]
    predicted_actions = predicted.get("actions", [predicted]) if isinstance(predicted, dict) else []

    exact_match = expected_actions == predicted_actions

    # field-level: only makes sense when action count matches
    field_results = []
    if len(expected_actions) == len(predicted_actions):
        for exp_act, pred_act in zip(expected_actions, predicted_actions):
            for field in ["action", "object", "target", "location", "quantity", "attributes"]:
                exp_val = exp_act.get(field)
                pred_val = pred_act.get(field) if isinstance(pred_act, dict) else None
                field_results.append(exp_val == pred_val)

    return {
        "exact_match": exact_match,
        "field_results": field_results,
    }


def compare_all(results: list):
    """Runs comparison on every result, adds exact_match + field_results."""
    logger.info("Comparing predictions to ground truth")

    for r in results:
        if not r["is_valid_json"]:
            r["exact_match"] = False
            r["field_results"] = []
            continue
        comparison = compare_single(r["expected"], r["predicted_parsed"])
        r["exact_match"] = comparison["exact_match"]
        r["field_results"] = comparison["field_results"]

    return results