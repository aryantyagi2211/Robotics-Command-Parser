import json
from datetime import datetime
from pathlib import Path
from src.logger import get_logger

logger = get_logger(__name__)

def save_eval_report(results: list, metrics: dict, output_dir: str = "reports"):
    """Saves evaluation results as a JSON report + a human-readable markdown summary."""
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- JSON report (full detail, machine-readable) ---
    json_path = f"{output_dir}/eval_report_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "metrics": metrics, "results": results}, f, indent=2, default=str)
    logger.info(f"Saved JSON report to {json_path}")

    # --- Markdown summary (human-readable) ---
    md_path = f"{output_dir}/eval_report_{timestamp}.md"
    md_lines = [
        f"# Evaluation Report — {timestamp}",
        "",
        "## Methodology",
        "- Held-out test set: 82 examples, never seen during training or validation",
        "- Deterministic generation (`do_sample=False`, greedy decoding)",
        "- JSON validity checked via balanced-brace extraction (handles trailing generation artifacts)",
        "- Exact-match: full predicted JSON must equal ground truth exactly",
        "- Field-level: each field compared independently",
        "- Simple/medium vs complex (multi-action) commands scored separately",
        "",
        "## Results",
        "",
        f"- **JSON validity**: {metrics['valid_json']}/{metrics['total']} ({metrics['valid_json_pct']:.1f}%)",
        f"- **Exact match**: {metrics['exact_match']}/{metrics['total']} ({metrics['exact_match_pct']:.1f}%)",
        "",
        "### Field-level accuracy",
        "",
        "| Field | Correct | Total | % |",
        "|---|---|---|---|",
    ]
    for field, (correct, total) in metrics["field_accuracy"].items():
        pct = 100 * correct / total if total else 0
        md_lines.append(f"| {field} | {correct} | {total} | {pct:.1f}% |")

    md_lines += [
        "",
        "### Simple vs complex commands",
        "",
        f"- Simple/medium: {metrics['simple_correct']}/{metrics['simple_total']} ({metrics['simple_pct']:.1f}%)",
        f"- Complex (multi-action): {metrics['complex_correct']}/{metrics['complex_total']} ({metrics['complex_pct']:.1f}%)",
    ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Saved markdown report to {md_path}")

    return json_path, md_path