"""
tests/test_auto_evaluator.py
-----------------------------
Unit tests for evaluators/auto_evaluator.py.

Run with:   python -m pytest tests/ -v
"""

import sys
import os

# Make sure the project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluators.auto_evaluator import (
    evaluate_response,
    evaluate_multiple,
    AutoEvaluationResult,
    CRITERION_WEIGHTS,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

PROMPT = "Explain the benefits of using Python for data science."

GOOD_RESPONSE = """
Python is widely used in data science for several reasons:

1. Ease of learning: Python has a clean syntax that reduces cognitive overhead.
2. Rich ecosystem: Libraries like NumPy, Pandas, and scikit-learn cover most needs.
3. Community: A large, active community means abundant tutorials and support.
4. Interoperability: Python integrates easily with SQL, R, and Java systems.

In summary, Python's combination of simplicity, power, and community support
makes it an excellent choice for data science projects of any scale.
"""

SHORT_RESPONSE = "Python is good for data."

EMPTY_RESPONSE = ""


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestEvaluateResponse:
    def test_returns_correct_type(self):
        result = evaluate_response(PROMPT, GOOD_RESPONSE)
        assert isinstance(result, AutoEvaluationResult)

    def test_all_scores_in_range(self):
        result = evaluate_response(PROMPT, GOOD_RESPONSE)
        for attr in ("accuracy", "completeness", "clarity", "creativity", "helpfulness"):
            score = getattr(result, attr).score
            assert 1.0 <= score <= 10.0, f"{attr} score {score} out of range"
        assert 1.0 <= result.overall_quality <= 10.0
        assert 1.0 <= result.weighted_score <= 10.0

    def test_good_response_beats_short_response(self):
        good  = evaluate_response(PROMPT, GOOD_RESPONSE)
        short = evaluate_response(PROMPT, SHORT_RESPONSE)
        # Completeness should clearly favour the longer response
        assert good.completeness.score > short.completeness.score

    def test_empty_response_scores_low(self):
        result = evaluate_response(PROMPT, EMPTY_RESPONSE)
        # Empty response should score near the floor on most criteria
        assert result.completeness.score <= 3.0

    def test_weighted_score_formula(self):
        result = evaluate_response(PROMPT, GOOD_RESPONSE)
        expected = round(
            result.accuracy.score * CRITERION_WEIGHTS["accuracy"] +
            result.completeness.score * CRITERION_WEIGHTS["completeness"] +
            result.clarity.score * CRITERION_WEIGHTS["clarity"] +
            result.creativity.score * CRITERION_WEIGHTS["creativity"] +
            result.helpfulness.score * CRITERION_WEIGHTS["helpfulness"],
            2
        )
        assert abs(result.weighted_score - expected) < 0.01

    def test_to_dict_keys(self):
        result = evaluate_response(PROMPT, GOOD_RESPONSE)
        d = result.to_dict()
        expected_keys = {
            "accuracy", "completeness", "clarity", "creativity",
            "helpfulness", "overall_quality", "weighted_score", "justification",
        }
        assert expected_keys == set(d.keys())

    def test_justifications_are_strings(self):
        result = evaluate_response(PROMPT, GOOD_RESPONSE)
        for attr in ("accuracy", "completeness", "clarity", "creativity", "helpfulness"):
            just = getattr(result, attr).justification
            assert isinstance(just, str) and len(just) > 0


class TestEvaluateMultiple:
    def test_returns_dict_keyed_by_model(self):
        responses = {
            "ModelA": GOOD_RESPONSE,
            "ModelB": SHORT_RESPONSE,
        }
        results = evaluate_multiple(PROMPT, responses)
        assert set(results.keys()) == {"ModelA", "ModelB"}

    def test_all_results_are_correct_type(self):
        responses = {"M1": GOOD_RESPONSE, "M2": SHORT_RESPONSE}
        for result in evaluate_multiple(PROMPT, responses).values():
            assert isinstance(result, AutoEvaluationResult)
