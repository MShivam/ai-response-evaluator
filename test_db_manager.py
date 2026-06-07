"""
tests/test_db_manager.py
-------------------------
Integration tests for database/db_manager.py.
Uses a temporary database file so production data is never touched.

Run with:   python -m pytest tests/ -v
"""

import sys
import os
import tempfile
import pathlib

# Patch DB_PATH before importing db_manager so tests use a temp file
_TMP_DIR = tempfile.mkdtemp()
_TMP_DB  = str(pathlib.Path(_TMP_DIR) / "test_evaluations.db")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Monkey-patch the path before importing
import database.db_manager as db_module
db_module.DB_PATH = pathlib.Path(_TMP_DB)  # type: ignore[attr-defined]

from database.db_manager import (
    initialize_db,
    create_evaluation,
    get_all_evaluations,
    search_evaluations,
    save_response,
    get_responses_for_evaluation,
    save_score,
    get_scores_for_response,
    get_full_evaluation_data,
)


# ─── Setup ───────────────────────────────────────────────────────────────────

def setup_module(_):
    initialize_db()


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestEvaluations:
    def test_create_and_fetch(self):
        eid = create_evaluation("What is AI?")
        assert isinstance(eid, int) and eid > 0

        evals = get_all_evaluations()
        prompts = [e["prompt"] for e in evals]
        assert "What is AI?" in prompts

    def test_search_evaluations(self):
        create_evaluation("Explain machine learning")
        results = search_evaluations("machine learning")
        assert any("machine learning" in r["prompt"] for r in results)

    def test_search_no_match(self):
        results = search_evaluations("zxqnotfound")
        assert results == []


class TestResponses:
    def test_save_and_fetch_response(self):
        eid  = create_evaluation("Test prompt for responses")
        rid  = save_response(eid, "gpt-4o", "A helpful answer.", tokens_used=50, latency_ms=120.5)
        assert isinstance(rid, int) and rid > 0

        resps = get_responses_for_evaluation(eid)
        assert len(resps) == 1
        assert resps[0]["model_name"] == "gpt-4o"
        assert resps[0]["response_text"] == "A helpful answer."

    def test_multiple_responses(self):
        eid = create_evaluation("Multi-model prompt")
        save_response(eid, "Model-A", "Response A")
        save_response(eid, "Model-B", "Response B")
        resps = get_responses_for_evaluation(eid)
        assert len(resps) == 2


class TestScores:
    def test_save_and_fetch_score(self):
        eid  = create_evaluation("Scoring test prompt")
        rid  = save_response(eid, "test-model", "Sample response")
        sid  = save_score(
            response_id=rid,
            scorer_type="human",
            accuracy=8.0, completeness=7.5, clarity=9.0,
            creativity=6.0, helpfulness=8.5,
            overall_quality=7.8, weighted_score=7.9,
            justification={"note": "Good overall"}
        )
        assert isinstance(sid, int)

        scores = get_scores_for_response(rid)
        assert len(scores) == 1
        assert scores[0]["scorer_type"] == "human"
        assert scores[0]["accuracy"] == 8.0
        assert isinstance(scores[0]["justification"], dict)


class TestFullEvaluationData:
    def test_full_data_structure(self):
        eid = create_evaluation("Full hydration test")
        rid = save_response(eid, "ModelX", "Some response text")
        save_score(
            rid, "automated", 7.0, 8.0, 6.5, 7.5, 8.0, 7.4, 7.4,
            justification={"accuracy": "Looks good"}
        )

        data = get_full_evaluation_data(eid)
        assert data["id"] == eid
        assert data["prompt"] == "Full hydration test"
        assert len(data["responses"]) == 1
        assert len(data["responses"][0]["scores"]) == 1

    def test_missing_evaluation_returns_empty(self):
        data = get_full_evaluation_data(999999)
        assert data == {}
