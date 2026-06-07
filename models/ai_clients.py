"""
models/ai_clients.py
---------------------
Abstracts communication with AI model APIs.

Supports:
  - Real OpenAI models (gpt-4o, gpt-4o-mini, gpt-3.5-turbo) when
    OPENAI_API_KEY is set in the environment.
  - A MockClient that generates deterministic fake responses so the app
    is fully functional without any API key.

All clients implement the same ModelClient protocol so the rest of the
application is agnostic to which backend is in use.
"""

import os
import time
import hashlib
import textwrap
import random
from typing import Protocol

# Try to import openai; fall back gracefully if not installed
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


# ─── Protocol (interface) ────────────────────────────────────────────────────

class ModelResponse:
    """Data-class holding everything we need from a model call."""
    def __init__(
        self,
        model_name: str,
        response_text: str,
        tokens_used: int = 0,
        latency_ms: float = 0.0,
        is_mock: bool = False,
    ):
        self.model_name = model_name
        self.response_text = response_text
        self.tokens_used = tokens_used
        self.latency_ms = latency_ms
        self.is_mock = is_mock

    def __repr__(self) -> str:
        return (
            f"ModelResponse(model={self.model_name!r}, "
            f"tokens={self.tokens_used}, latency={self.latency_ms:.0f}ms)"
        )


class ModelClient(Protocol):
    """Any object that can generate a response for a prompt."""
    def generate(self, prompt: str, system_prompt: str = "") -> ModelResponse:
        ...


# ─── OpenAI Client ───────────────────────────────────────────────────────────

class OpenAIClient:
    """
    Thin wrapper around the official OpenAI Python SDK.
    Raises RuntimeError if the SDK is unavailable or no API key is found.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        if not _OPENAI_AVAILABLE:
            raise RuntimeError("openai package is not installed.")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set.")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, system_prompt: str = "") -> ModelResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=1024,
            temperature=0.7,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        text = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0
        return ModelResponse(
            model_name=self.model_name,
            response_text=text,
            tokens_used=tokens,
            latency_ms=latency_ms,
            is_mock=False,
        )


# ─── Mock Client ─────────────────────────────────────────────────────────────

# A bank of varied template paragraphs so each model "sounds" different
_MOCK_TEMPLATES: dict[str, list[str]] = {
    "Mock-GPT-4o": [
        (
            "As an advanced language model, I'll provide a comprehensive answer. "
            "{topic} is a multifaceted subject that spans several domains. "
            "Key considerations include: (1) foundational theory, which underpins the concept; "
            "(2) practical application, which determines real-world utility; "
            "and (3) emerging trends, which shape future developments. "
            "In summary, a nuanced understanding of {topic} requires balancing "
            "theoretical rigor with pragmatic insight."
        ),
        (
            "Excellent question about {topic}. Let me break this down systematically. "
            "First, the historical context shows how {topic} evolved over time. "
            "Second, current best practices suggest a structured approach. "
            "Third, potential pitfalls should be carefully managed. "
            "Overall, success with {topic} depends on clear goals, iterative feedback, "
            "and domain expertise."
        ),
    ],
    "Mock-GPT-4o-mini": [
        (
            "Sure! {topic} is really interesting. Here are the main points: "
            "• It has clear benefits when applied correctly. "
            "• You need the right context and tools. "
            "• Common mistakes include rushing and skipping fundamentals. "
            "Hope this helps!"
        ),
        (
            "Great question! Regarding {topic}: the core idea is straightforward. "
            "Start with the basics, build gradually, and test often. "
            "The most important thing is to stay consistent and learn from feedback. "
            "Let me know if you want more detail on any part!"
        ),
    ],
    "Mock-Claude-3": [
        (
            "I'd be happy to discuss {topic} thoughtfully. "
            "This is an area with interesting nuances worth exploring carefully. "
            "From a principled standpoint, the most important factors are accuracy, "
            "clarity of reasoning, and intellectual honesty about uncertainty. "
            "I'll note that reasonable people can disagree on some aspects of {topic}, "
            "and I think that epistemic humility is valuable here. "
            "That said, the preponderance of evidence suggests a measured, evidence-based approach."
        ),
    ],
    "Mock-Gemini-Pro": [
        (
            "Thinking about {topic} from multiple angles: "
            "The technical perspective emphasizes structured problem-solving and data. "
            "The creative perspective encourages lateral thinking and experimentation. "
            "The practical perspective focuses on deliverables and timelines. "
            "Synthesising these views, a robust strategy for {topic} combines "
            "analytical rigour with creative flexibility, always anchored in measurable outcomes."
        ),
    ],
}

_DEFAULT_TEMPLATES = [
    "This is a simulated response about {topic}. "
    "The model would normally provide a detailed answer here, "
    "drawing on its training data and reasoning capabilities. "
    "Key points would include context, analysis, and actionable insights."
]


class MockClient:
    """
    Generates deterministic-ish fake responses for testing and demo purposes.
    Responses vary by model name and prompt content so comparisons look realistic.
    No API key or network required.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str, system_prompt: str = "") -> ModelResponse:  # noqa: ARG002
        # Simulate network latency (50–400 ms) reproducibly
        seed = int(hashlib.md5(f"{self.model_name}{prompt}".encode()).hexdigest(), 16)
        rng = random.Random(seed)
        latency_ms = rng.uniform(50, 400)
        time.sleep(latency_ms / 1000)  # Simulate the wait

        templates = _MOCK_TEMPLATES.get(self.model_name, _DEFAULT_TEMPLATES)
        template = templates[seed % len(templates)]

        # Extract a rough topic from the first 5 words of the prompt
        words = prompt.split()
        topic = " ".join(words[:5]) if len(words) >= 5 else prompt
        response_text = template.format(topic=topic)

        # Simulate token count (≈ 4 chars per token)
        tokens_used = len(response_text) // 4

        return ModelResponse(
            model_name=self.model_name,
            response_text=response_text,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            is_mock=True,
        )


# ─── Registry & Factory ──────────────────────────────────────────────────────

# Models available in the UI. Key = display name, value = OpenAI model id (if real)
AVAILABLE_MODELS: dict[str, str | None] = {
    "Mock-GPT-4o": None,
    "Mock-GPT-4o-mini": None,
    "Mock-Claude-3": None,
    "Mock-Gemini-Pro": None,
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
}

# Models that are always mock (for UI labelling)
MOCK_MODEL_NAMES = {k for k, v in AVAILABLE_MODELS.items() if v is None}


def get_client(model_name: str) -> ModelClient:
    """
    Factory: returns a real OpenAIClient if the API key is set and the
    model is a real OpenAI model; otherwise returns a MockClient.
    """
    openai_id = AVAILABLE_MODELS.get(model_name)
    has_key = bool(os.environ.get("OPENAI_API_KEY"))

    if openai_id and has_key and _OPENAI_AVAILABLE:
        try:
            return OpenAIClient(model_name=openai_id)
        except RuntimeError:
            pass  # Fall through to mock

    return MockClient(model_name=model_name)
