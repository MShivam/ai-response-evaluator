"""
utils/export_utils.py
----------------------
Functions to serialise evaluation data into CSV and JSON formats.
Exports are written to the /exports directory and the raw bytes are
also returned so Streamlit's st.download_button can serve them directly.
"""

import csv
import json
import io
from datetime import datetime
from pathlib import Path
from typing import Any

EXPORTS_DIR = Path(__file__).parent.parent / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ─── CSV Export ──────────────────────────────────────────────────────────────

def export_to_csv(evaluation_data: dict[str, Any]) -> tuple[bytes, str]:
    """
    Flatten one fully-hydrated evaluation dict into a CSV.

    Returns:
        (csv_bytes, filename)  – ready for st.download_button
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header row
    writer.writerow([
        "evaluation_id", "prompt", "created_at",
        "model_name", "response_text", "tokens_used", "latency_ms",
        "scorer_type",
        "accuracy", "completeness", "clarity",
        "creativity", "helpfulness", "overall_quality", "weighted_score",
        "justification",
    ])

    eval_id    = evaluation_data.get("id", "")
    prompt     = evaluation_data.get("prompt", "")
    created_at = evaluation_data.get("created_at", "")

    for resp in evaluation_data.get("responses", []):
        for score in resp.get("scores", []):
            just = score.get("justification", "")
            if isinstance(just, dict):
                just = json.dumps(just)
            writer.writerow([
                eval_id, prompt, created_at,
                resp["model_name"], resp["response_text"],
                resp.get("tokens_used", 0), resp.get("latency_ms", 0),
                score.get("scorer_type", ""),
                score.get("accuracy", ""),
                score.get("completeness", ""),
                score.get("clarity", ""),
                score.get("creativity", ""),
                score.get("helpfulness", ""),
                score.get("overall_quality", ""),
                score.get("weighted_score", ""),
                just,
            ])

    csv_bytes = output.getvalue().encode("utf-8")
    filename  = f"evaluation_{eval_id}_{_timestamp()}.csv"

    # Also save to disk
    (EXPORTS_DIR / filename).write_bytes(csv_bytes)
    return csv_bytes, filename


def export_all_to_csv(evaluations: list[dict[str, Any]]) -> tuple[bytes, str]:
    """Export multiple evaluations into one combined CSV."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "evaluation_id", "prompt", "created_at",
        "model_name", "response_text", "tokens_used", "latency_ms",
        "scorer_type",
        "accuracy", "completeness", "clarity",
        "creativity", "helpfulness", "overall_quality", "weighted_score",
    ])

    for ev in evaluations:
        for resp in ev.get("responses", []):
            for score in resp.get("scores", []):
                writer.writerow([
                    ev.get("id", ""), ev.get("prompt", ""), ev.get("created_at", ""),
                    resp["model_name"], resp["response_text"],
                    resp.get("tokens_used", 0), resp.get("latency_ms", 0),
                    score.get("scorer_type", ""),
                    score.get("accuracy", ""),
                    score.get("completeness", ""),
                    score.get("clarity", ""),
                    score.get("creativity", ""),
                    score.get("helpfulness", ""),
                    score.get("overall_quality", ""),
                    score.get("weighted_score", ""),
                ])

    csv_bytes = output.getvalue().encode("utf-8")
    filename  = f"all_evaluations_{_timestamp()}.csv"
    (EXPORTS_DIR / filename).write_bytes(csv_bytes)
    return csv_bytes, filename


# ─── JSON Export ─────────────────────────────────────────────────────────────

def export_to_json(evaluation_data: dict[str, Any]) -> tuple[bytes, str]:
    """
    Serialise a single evaluation to pretty-printed JSON.

    Returns:
        (json_bytes, filename)
    """
    json_bytes = json.dumps(evaluation_data, indent=2, default=str).encode("utf-8")
    eval_id    = evaluation_data.get("id", "unknown")
    filename   = f"evaluation_{eval_id}_{_timestamp()}.json"
    (EXPORTS_DIR / filename).write_bytes(json_bytes)
    return json_bytes, filename


def export_all_to_json(evaluations: list[dict[str, Any]]) -> tuple[bytes, str]:
    """Export a list of evaluations to a single JSON file."""
    json_bytes = json.dumps(evaluations, indent=2, default=str).encode("utf-8")
    filename   = f"all_evaluations_{_timestamp()}.json"
    (EXPORTS_DIR / filename).write_bytes(json_bytes)
    return json_bytes, filename
