"""
evaluators/auto_evaluator.py
-----------------------------
Automated evaluation engine that scores AI responses on six criteria
without requiring human input.

Scoring philosophy
------------------
Rather than calling another LLM to judge responses (which adds cost and
latency), this module uses a suite of fast heuristic signals:

  Accuracy       – coherence proxies: sentence variety, hedge word ratio
  Completeness   – length relative to prompt, structured element presence
  Clarity        – avg sentence length, punctuation density, reading ease
  Creativity     – lexical diversity (type-token ratio), unusual bigrams
  Helpfulness    – action words, numbered steps, presence of examples
  Overall Quality – weighted composite of the above five

Each criterion returns a float on [1, 10] plus a plain-English justification.

This approach is intentionally transparent and auditable — no black-box
model calls required.
"""

import re
import math
import string
from dataclasses import dataclass, field
from typing import Optional

# ─── Weights for the composite score ────────────────────────────────────────
CRITERION_WEIGHTS: dict[str, float] = {
    "accuracy":        0.25,
    "completeness":    0.20,
    "clarity":         0.20,
    "creativity":      0.15,
    "helpfulness":     0.20,
}

# ─── Thresholds & constants ──────────────────────────────────────────────────
HEDGE_WORDS = {
    "perhaps", "maybe", "might", "could", "possibly", "approximately",
    "roughly", "generally", "often", "sometimes", "usually", "likely",
}

ACTION_WORDS = {
    "use", "create", "build", "implement", "run", "check", "install",
    "configure", "ensure", "note", "consider", "remember", "try",
    "start", "stop", "open", "close", "set", "get", "add", "remove",
}


@dataclass
class CriterionResult:
    """Score + explanation for a single evaluation dimension."""
    score: float        # 1.0 – 10.0
    justification: str


@dataclass
class AutoEvaluationResult:
    """Complete automated evaluation for one response."""
    accuracy: CriterionResult = field(default_factory=lambda: CriterionResult(0.0, ""))
    completeness: CriterionResult = field(default_factory=lambda: CriterionResult(0.0, ""))
    clarity: CriterionResult = field(default_factory=lambda: CriterionResult(0.0, ""))
    creativity: CriterionResult = field(default_factory=lambda: CriterionResult(0.0, ""))
    helpfulness: CriterionResult = field(default_factory=lambda: CriterionResult(0.0, ""))
    overall_quality: float = 0.0
    weighted_score: float = 0.0

    def to_dict(self) -> dict:
        """Flatten to a plain dict for database storage and display."""
        return {
            "accuracy":        self.accuracy.score,
            "completeness":    self.completeness.score,
            "clarity":         self.clarity.score,
            "creativity":      self.creativity.score,
            "helpfulness":     self.helpfulness.score,
            "overall_quality": self.overall_quality,
            "weighted_score":  self.weighted_score,
            "justification": {
                "accuracy":     self.accuracy.justification,
                "completeness": self.completeness.justification,
                "clarity":      self.clarity.justification,
                "creativity":   self.creativity.justification,
                "helpfulness":  self.helpfulness.justification,
            },
        }


# ─── Helper utilities ────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Split on whitespace, strip punctuation, lowercase."""
    return [
        w.strip(string.punctuation).lower()
        for w in text.split()
        if w.strip(string.punctuation)
    ]


def _sentences(text: str) -> list[str]:
    """Naive sentence splitter on . ! ?"""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p]


def _clamp(value: float, lo: float = 1.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, value))


def _scale(raw: float, raw_min: float, raw_max: float) -> float:
    """Map raw ∈ [raw_min, raw_max] → [1, 10]."""
    if raw_max == raw_min:
        return 5.0
    normalized = (raw - raw_min) / (raw_max - raw_min)
    return _clamp(1.0 + normalized * 9.0)


# ─── Criterion scorers ───────────────────────────────────────────────────────

def _score_accuracy(response: str, prompt: str) -> CriterionResult:
    """
    Proxy for accuracy: measures how topically relevant the response is
    using prompt-word coverage and penalises excessive hedging.
    """
    prompt_words = set(_tokenize(prompt)) - {"the", "a", "an", "is", "are", "what", "how"}
    resp_words   = set(_tokenize(response))

    # Coverage: fraction of meaningful prompt words that appear in response
    coverage = len(prompt_words & resp_words) / max(len(prompt_words), 1)

    # Hedge penalty: high hedge ratio suggests low confidence / hand-waving
    all_resp_words = _tokenize(response)
    hedge_ratio = sum(1 for w in all_resp_words if w in HEDGE_WORDS) / max(len(all_resp_words), 1)
    hedge_penalty = min(hedge_ratio * 20, 2.0)   # up to −2 points

    raw_score = _clamp(coverage * 10 - hedge_penalty)

    if coverage >= 0.7:
        reason = f"Response covers {coverage:.0%} of prompt concepts with good topical relevance."
    elif coverage >= 0.4:
        reason = f"Response covers {coverage:.0%} of prompt concepts; some gaps present."
    else:
        reason = f"Low coverage ({coverage:.0%}) of prompt concepts; may be off-topic."

    if hedge_ratio > 0.05:
        reason += f" Hedge word ratio ({hedge_ratio:.1%}) reduces confidence score."

    return CriterionResult(score=round(raw_score, 2), justification=reason)


def _score_completeness(response: str, prompt: str) -> CriterionResult:
    """
    Measures depth: word count relative to prompt length, presence of
    structure (bullets, numbered lists, paragraphs).
    """
    resp_words   = _tokenize(response)
    prompt_words = _tokenize(prompt)
    word_count   = len(resp_words)
    prompt_len   = max(len(prompt_words), 1)

    # Responses should be substantially longer than the prompt
    length_ratio = word_count / prompt_len

    # Structural richness: bullets, numbers, headers
    has_bullets  = bool(re.search(r'[•\-\*]\s', response))
    has_numbered = bool(re.search(r'\d+[.)]\s', response))
    has_headers  = bool(re.search(r'\n[A-Z][^\n]{0,40}:\n', response))
    structure_bonus = sum([has_bullets, has_numbered, has_headers]) * 0.5

    # Map length ratio: ideal is 10–50× the prompt
    length_score = _scale(math.log1p(length_ratio), 0, math.log1p(50))
    raw_score    = _clamp(length_score + structure_bonus)

    justification = (
        f"Response contains {word_count} words "
        f"({length_ratio:.1f}× prompt length). "
    )
    if has_bullets or has_numbered:
        justification += "Structured formatting (lists) detected, improving coverage signal. "
    if word_count < 30:
        justification += "Very short — likely incomplete."
    elif word_count > 200:
        justification += "Detailed response indicates good coverage."

    return CriterionResult(score=round(raw_score, 2), justification=justification)


def _score_clarity(response: str, _prompt: str) -> CriterionResult:
    """
    Assesses readability: average sentence length, punctuation density,
    and absence of run-on sentences.
    """
    sents = _sentences(response)
    if not sents:
        return CriterionResult(score=1.0, justification="Empty response.")

    words_per_sent = [len(_tokenize(s)) for s in sents]
    avg_wps = sum(words_per_sent) / len(words_per_sent)

    # Ideal sentence length: 12–20 words (penalty for very long/short)
    if 12 <= avg_wps <= 20:
        length_score = 10.0
    elif avg_wps < 5:
        length_score = _scale(avg_wps, 1, 12)
    else:
        length_score = _scale(1 / max(avg_wps - 20, 1), 0, 1)
        length_score = max(4.0, 10.0 - (avg_wps - 20) * 0.15)

    # Punctuation density as a readability signal (too little = wall of text)
    punct_count  = sum(1 for c in response if c in string.punctuation)
    total_chars  = max(len(response), 1)
    punct_ratio  = punct_count / total_chars
    punct_score  = _scale(punct_ratio, 0.02, 0.15)

    raw_score = _clamp((length_score * 0.7) + (punct_score * 0.3))
    justification = (
        f"Average sentence length: {avg_wps:.1f} words "
        f"({'ideal' if 12 <= avg_wps <= 20 else 'sub-optimal'}). "
        f"Punctuation density: {punct_ratio:.2%}."
    )
    return CriterionResult(score=round(raw_score, 2), justification=justification)


def _score_creativity(response: str, _prompt: str) -> CriterionResult:
    """
    Estimates lexical diversity via type-token ratio (TTR) and detects
    uncommon word usage as a creativity proxy.
    """
    tokens = _tokenize(response)
    if not tokens:
        return CriterionResult(score=1.0, justification="Empty response.")

    unique_tokens = set(tokens)
    ttr = len(unique_tokens) / len(tokens)

    # Common filler words reduce diversity value
    common_words = {
        "the", "a", "an", "is", "are", "was", "were", "and", "or", "but",
        "in", "on", "at", "to", "for", "of", "it", "this", "that", "with",
    }
    meaningful_unique = unique_tokens - common_words
    meaningful_ratio  = len(meaningful_unique) / max(len(unique_tokens), 1)

    raw_score = _clamp(ttr * 7 + meaningful_ratio * 3)
    justification = (
        f"Type-token ratio: {ttr:.2f} "
        f"({len(unique_tokens)} unique / {len(tokens)} total tokens). "
        f"Meaningful vocabulary ratio: {meaningful_ratio:.2%}."
    )
    return CriterionResult(score=round(raw_score, 2), justification=justification)


def _score_helpfulness(response: str, prompt: str) -> CriterionResult:
    """
    Measures actionability: presence of action verbs, numbered steps,
    examples, and code blocks.
    """
    lower = response.lower()
    tokens = _tokenize(response)

    action_count = sum(1 for w in tokens if w in ACTION_WORDS)
    action_density = action_count / max(len(tokens), 1)

    has_steps    = bool(re.search(r'\d+[.)]\s', response))
    has_example  = bool(re.search(r'\bexample\b|\be\.g\.\b|\bfor instance\b', lower))
    has_code     = bool(re.search(r'`[^`]+`|```', response))
    has_summary  = bool(re.search(r'\bin summary\b|\bto summarize\b|\boverall\b', lower))

    bonus = sum([has_steps, has_example, has_code, has_summary]) * 0.75
    raw_score = _clamp(action_density * 30 + bonus + 4.0)

    features = []
    if has_steps:    features.append("numbered steps")
    if has_example:  features.append("examples")
    if has_code:     features.append("code blocks")
    if has_summary:  features.append("summary")

    justification = (
        f"Action word density: {action_density:.2%}. "
        + (f"Helpful features: {', '.join(features)}." if features else "No explicit structure found.")
    )
    return CriterionResult(score=round(raw_score, 2), justification=justification)


# ─── Public API ──────────────────────────────────────────────────────────────

def evaluate_response(prompt: str, response: str) -> AutoEvaluationResult:
    """
    Run all heuristic scorers on a single (prompt, response) pair and
    return a fully populated AutoEvaluationResult.
    """
    result = AutoEvaluationResult()

    result.accuracy     = _score_accuracy(response, prompt)
    result.completeness = _score_completeness(response, prompt)
    result.clarity      = _score_clarity(response, prompt)
    result.creativity   = _score_creativity(response, prompt)
    result.helpfulness  = _score_helpfulness(response, prompt)

    # Weighted composite
    scores = {
        "accuracy":     result.accuracy.score,
        "completeness": result.completeness.score,
        "clarity":      result.clarity.score,
        "creativity":   result.creativity.score,
        "helpfulness":  result.helpfulness.score,
    }
    result.weighted_score = round(
        sum(scores[k] * CRITERION_WEIGHTS[k] for k in scores), 2
    )

    # Simple average for "overall quality" (unweighted)
    result.overall_quality = round(sum(scores.values()) / len(scores), 2)

    return result


def evaluate_multiple(
    prompt: str,
    responses: dict[str, str],
) -> dict[str, AutoEvaluationResult]:
    """
    Convenience wrapper: evaluate all responses in a {model_name: text} dict.
    Returns {model_name: AutoEvaluationResult}.
    """
    return {model: evaluate_response(prompt, text) for model, text in responses.items()}
