"""Curated model list shown in the dropdown.

Edit this file to add/remove models. Model IDs come from
https://openrouter.ai/models — copy the exact slug.

Pricing is approximate per 1M tokens (input / output) for guidance only.
OpenRouter pricing changes frequently; check the URL above for the truth.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    id: str
    label: str
    note: str  # short description shown next to the option


MODELS: tuple[Model, ...] = (
    Model(
        id="deepseek/deepseek-chat-v3.1:free",
        label="DeepSeek V3.1 (free)",
        note="Free · solid quality · rate-limited (~50/day without credits)",
    ),
    Model(
        id="meta-llama/llama-3.3-70b-instruct:free",
        label="Llama 3.3 70B (free)",
        note="Free · alternative to DeepSeek if it's rate-limited",
    ),
    Model(
        id="google/gemini-2.0-flash-exp:free",
        label="Gemini 2.0 Flash (free)",
        note="Free · fast · sometimes flaky / unavailable",
    ),
    Model(
        id="deepseek/deepseek-chat-v3.1",
        label="DeepSeek V3.1 (paid)",
        note="~$0.27 / 1M input · highest free-tier-equivalent quality",
    ),
    Model(
        id="anthropic/claude-3.5-haiku",
        label="Claude 3.5 Haiku",
        note="~$0.80 / 1M input · fast · good format adherence",
    ),
    Model(
        id="anthropic/claude-sonnet-4.5",
        label="Claude Sonnet 4.5",
        note="~$3 / 1M input · best overall quality",
    ),
    Model(
        id="openai/gpt-4o-mini",
        label="GPT-4o mini",
        note="~$0.15 / 1M input · cheap · solid quality",
    ),
)

DEFAULT_MODEL_ID: str = MODELS[0].id


def is_known(model_id: str) -> bool:
    return any(m.id == model_id for m in MODELS)


def get(model_id: str) -> Model | None:
    for m in MODELS:
        if m.id == model_id:
            return m
    return None
