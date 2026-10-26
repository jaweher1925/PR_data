"""RQ1 LLM-as-judge engine: the model-calling primitives shared by every run.

``complete_judgment``/``judge_case`` make one forced-tool-use judgment through
LiteLLM; ``complete_text`` is a plain completion for insight distillation; the
raters wrap providers behind one interface; ``load_cases`` reads the gold case
file; ``_kappa``/``_binary`` back the agreement report. Requires
``ANTHROPIC_API_KEY`` (env or ``.env``) for any live call.
"""
from __future__ import annotations

import json
import os
from typing import Protocol, runtime_checkable

import litellm

from prfix.config import JUDGE_DIR, PROJECT_ROOT
from prfix.rq1_fix_rates.judge.prompt import JUDGE_TOOL, coerce_label, render_case

# drop_params=False: an unsupported (model, param) pair must error, not vanish --
# the judge is reproducibility-sensitive. suppress_debug_info quiets LiteLLM's
# per-exception "Give Feedback / Get help" banner.
litellm.drop_params = False
litellm.suppress_debug_info = True

# Per-request timeout + retries: a hung call would otherwise stall the thread
# pool's final as_completed future forever. A stuck/transient call fails fast and
# recovers; a persistent failure surfaces as an ERROR row (excluded, re-judged on
# the next resumable run). Tune via PRFIX_JUDGE_TIMEOUT (seconds).
JUDGE_TIMEOUT_S = float(os.environ.get("PRFIX_JUDGE_TIMEOUT", "180"))
JUDGE_NUM_RETRIES = 2
_EFFORT_FIELD = 2  # index of the optional effort part in provider:model[:effort]

# JUDGE_TOOL as an OpenAI-style function tool; LiteLLM converts it to Anthropic's
# tool_use on the wire, so one definition serves both providers (response comes
# back OpenAI-shaped either way).
_JUDGE_FUNCTION_TOOL = {
    "type": "function",
    "function": {
        "name": JUDGE_TOOL["name"],
        "description": JUDGE_TOOL["description"],
        "parameters": JUDGE_TOOL["input_schema"],
    },
}

CASES_JSON = JUDGE_DIR / "rq1_judge_cases.json"

# Models that 400 on sampling params; only Sonnet 4.6 / Haiku still accept them.
_NO_SAMPLING = ("opus-4-7", "opus-4-8", "fable-5", "mythos-5", "mythos-preview")


def _supports_temperature(model: str) -> bool:
    return not any(tag in model for tag in _NO_SAMPLING)


def _resolve_env(name: str) -> str | None:
    """``name`` from the environment, else from .env (project convention)."""
    val = os.environ.get(name)
    if val:
        return val
    from prfix.common.loaders.github_api import _load_dotenv  # noqa: PLC0415
    return _load_dotenv(PROJECT_ROOT / ".env").get(name)


def _resolve_api_key() -> str | None:
    """ANTHROPIC_API_KEY from the environment, else from .env."""
    return _resolve_env("ANTHROPIC_API_KEY")


def _parse_tool_result(resp) -> dict:
    """Map a LiteLLM (OpenAI-shaped) response into the judge's result dict.

    Canonicalises the label (forced tool-use doesn't hard-guarantee the enum) and
    distinguishes a length-truncated response (``finish_reason == 'length'``) from
    a genuine no-decision, so a cut-off trace becomes an ERROR, not a silent
    "Indeterminate".
    """
    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        out = dict(json.loads(tool_calls[0].function.arguments))
        out["evidence_files"] = out.get("evidence_files", [])
        raw = out.get("label")
        canon = coerce_label(raw)
        if canon is None:
            out["label"] = "ERROR"
            out["rationale"] = f"off-enum label {raw!r}: {out.get('rationale', '')}"
        else:
            out["label"] = canon
        return out
    if resp.choices[0].finish_reason == "length":
        return {"label": "ERROR", "confidence": 0.0,
                "rationale": "response truncated (finish_reason=length) before tool call",
                "evidence_files": []}
    return {"label": "Indeterminate", "confidence": 0.0,
            "rationale": "no tool call returned", "evidence_files": []}


def complete_judgment(model: str, case: dict, system_prompt: str, *, api_key: str,
                      api_base: str | None = None, max_tokens: int | None = None,
                      temperature: float | None = None, cache_system: bool = False,
                      extra: dict | None = None) -> dict:
    """One judge call through LiteLLM -> {label, confidence, rationale, evidence_files}.

    ``model`` is a LiteLLM id (``anthropic/<m>`` or ``openai/<m>``); the forced
    ``submit_judgment`` tool is reused for both providers. ``cache_system`` sends
    the system prompt as an Anthropic ephemeral-cache block (Anthropic only;
    OpenAI gets a plain string). ``extra`` carries provider params LiteLLM
    forwards untouched (``output_config`` for Anthropic effort,
    ``reasoning_effort`` for OpenAI). ``api_base`` points at a custom gateway.
    """
    system_msg = (
        {"role": "system",
         "content": [{"type": "text", "text": system_prompt,
                      "cache_control": {"type": "ephemeral"}}]}
        if cache_system else
        {"role": "system", "content": system_prompt}
    )
    kwargs: dict = dict(
        model=model,
        messages=[system_msg, {"role": "user", "content": render_case(case)}],
        tools=[_JUDGE_FUNCTION_TOOL],
        tool_choice={"type": "function", "function": {"name": JUDGE_TOOL["name"]}},
        api_key=api_key,
        timeout=JUDGE_TIMEOUT_S,       # fail a hung call instead of stalling the pool
        num_retries=JUDGE_NUM_RETRIES,  # ride out a transient stall / drop
    )
    if api_base:
        kwargs["api_base"] = api_base
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if extra:
        kwargs.update(extra)
    return _parse_tool_result(litellm.completion(**kwargs))


def complete_text(model: str, system: str, user: str, *, api_key: str,
                  api_base: str | None = None, max_tokens: int = 4096,
                  temperature: float | None = None, extra: dict | None = None) -> str:
    """A plain (no-tool) LiteLLM completion returning the assistant's text.

    Used by insight distillation (``prompt.distil``). ``model`` is a LiteLLM id
    (``anthropic/<m>``); ``extra`` carries provider params LiteLLM forwards
    untouched (e.g. ``output_config`` for Anthropic effort)."""
    kwargs: dict = dict(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        api_key=api_key,
        timeout=JUDGE_TIMEOUT_S,
        num_retries=JUDGE_NUM_RETRIES,
        max_tokens=max_tokens,
    )
    if api_base:
        kwargs["api_base"] = api_base
    if temperature is not None:
        kwargs["temperature"] = temperature
    if extra:
        kwargs.update(extra)
    return litellm.completion(**kwargs).choices[0].message.content or ""


def load_cases() -> list[dict]:
    if not CASES_JSON.exists():
        raise FileNotFoundError(
            f"{CASES_JSON} not found -- build it: uv run python -m "
            f"prfix.rq1_fix_rates.judge.build_cases --mode gold"
        )
    return json.loads(CASES_JSON.read_text())


def judge_case(case: dict, system_prompt: str, model: str, effort: str | None = None,
               *, api_key: str, api_base: str | None = None) -> dict:
    """One Claude judgment via LiteLLM (Anthropic provider) -> {label, ...}.

    Preserves the native request: ``temperature=0`` where accepted
    (bit-reproducible), the system prompt as an ephemeral cache block, forced
    ``submit_judgment`` tool-use, and ``effort`` = ``output_config.effort``
    (``None`` = model default). ``max_tokens=8096`` gives a high-effort reasoning
    trace headroom so it can't exhaust the budget before the tool block is emitted
    (truncation would otherwise become a silent "Indeterminate"). Warm one call
    before fanning out so the cached system prompt is written first.
    """
    return complete_judgment(
        f"anthropic/{model}", case, system_prompt,
        api_key=api_key, api_base=api_base, max_tokens=8096,
        temperature=0 if _supports_temperature(model) else None,
        cache_system=True,
        extra={"output_config": {"effort": effort}} if effort else None,
    )


def _binary(label: str, mode: str) -> int:
    """strict: Direct fix == 1.  lenient: Direct fix | Related touch == 1."""
    if mode == "strict":
        return int(label == "Direct fix")
    return int(label in ("Direct fix", "Related touch"))


def _kappa(a: list, b: list) -> float:
    """Cohen's kappa, robust to a degenerate single-class column (returns nan)."""
    from sklearn.metrics import cohen_kappa_score  # noqa: PLC0415

    constant = lambda xs: all(x == xs[0] for x in xs)  # noqa: E731
    if not a or (constant(a) and constant(b)):
        return float("nan")  # empty or both single-class -> undefined
    return float(cohen_kappa_score(a, b))


# ============================================================================
# Raters -- provider-agnostic wrappers for the population run (run_judge.py); each
# returns {label, confidence, rationale, evidence_files} so prompt/scoring code
# never changes between models.
# ============================================================================

# reasoning_effort values OpenAI documents; support varies by model, so we
# validate against this set and let the API reject unsupported pairs.
OPENAI_EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh"]


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text).strip("-")


def _resolve_key(name: str) -> str | None:
    """``name`` from the environment, else from ``.env`` (project convention)."""
    key = os.environ.get(name)
    if key:
        return key
    from prfix.common.loaders.github_api import _load_dotenv  # noqa: PLC0415
    return _load_dotenv(PROJECT_ROOT / ".env").get(name)


@runtime_checkable
class Rater(Protocol):
    provider: str
    model: str

    def judge_case(self, case: dict, system_prompt: str) -> dict: ...


class AnthropicRater:
    """Claude via forced tool-use, dispatched through LiteLLM (``engine.judge_case``).

    ``model`` is passed through as-is: a real Anthropic model id against the
    public API, or whatever alias a custom gateway (``ANTHROPIC_BASE_URL``)
    exposes."""

    provider = "anthropic"

    def __init__(self, model: str, effort: str | None = None):
        self.model = model
        self.effort = effort
        key = _resolve_key("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("ANTHROPIC_API_KEY not set (env or .env).")
        self._key = key
        # Optional custom gateway via ANTHROPIC_BASE_URL (env or .env).
        self._base_url = _resolve_key("ANTHROPIC_BASE_URL")

    def judge_case(self, case: dict, system_prompt: str) -> dict:
        return judge_case(case, system_prompt, self.model, self.effort,
                          api_key=self._key, api_base=self._base_url)


class OpenAIRater:
    """GPT via LiteLLM forced function-calling (Chat Completions).

    Reuses the shared ``submit_judgment`` tool, steers depth with
    ``reasoning_effort`` (see ``OPENAI_EFFORTS``; ``None``/"" omits it), and omits
    ``temperature`` (reasoning models reject sampling params). ``model`` is passed
    through as-is; set ``OPENAI_BASE_URL`` to route through an OpenAI-compatible
    gateway.
    """

    provider = "openai"

    def __init__(self, model: str, reasoning_effort: str | None = None):
        self.model = model
        if reasoning_effort in ("", "default", None):
            reasoning_effort = None
        elif reasoning_effort not in OPENAI_EFFORTS:
            raise ValueError(
                f"reasoning_effort={reasoning_effort!r} invalid; choose from "
                f"{OPENAI_EFFORTS} (or '' for the model default).")
        self.reasoning_effort = reasoning_effort
        key = _resolve_key("OPENAI_API_KEY")
        if not key:
            raise SystemExit(
                "OPENAI_API_KEY not set. Add a line `OPENAI_API_KEY=sk-...` to "
                ".env (ANTHROPIC_API_KEY lives there too) or export it.")
        self._key = key
        self.effort = self.reasoning_effort  # uniform name for file identifiers
        # Optional OpenAI-compatible gateway via OPENAI_BASE_URL (env or .env).
        self._base_url = _resolve_key("OPENAI_BASE_URL")

    def judge_case(self, case: dict, system_prompt: str) -> dict:
        # No max_tokens and no cached system block -- both match the original
        # OpenAI Chat Completions call.
        return complete_judgment(
            f"openai/{self.model}", case, system_prompt,
            api_key=self._key, api_base=self._base_url,
            max_tokens=None, temperature=None, cache_system=False,
            extra={"reasoning_effort": self.reasoning_effort}
            if self.reasoning_effort else None,
        )


def parse_spec(spec: str) -> tuple[str, str, str | None]:
    """Split a ``provider:model[:effort]`` spec into its parts WITHOUT building
    a rater, so no API key is needed. The checkpoint and CSV identifiers derive
    from exactly these three values, which is what a key-free consolidation of
    an already-recorded run needs."""
    parts = spec.split(":", 2)
    provider = parts[0].lower()
    model = parts[1] if len(parts) > 1 and parts[1] else None
    # None => send no effort param.
    effort = parts[_EFFORT_FIELD] if len(parts) > _EFFORT_FIELD else None
    if not model:
        raise SystemExit(
            f"rater spec {spec!r} has no model -- use provider:model[:effort], "
            f"e.g. anthropic:claude-opus-4-8:low")
    if provider == "openai" and effort in ("", "default"):
        effort = None
    if provider not in ("anthropic", "openai"):
        raise ValueError(f"unknown provider {provider!r} (use 'anthropic' or 'openai')")
    return provider, model, effort


def make_rater(spec: str) -> Rater:
    """Build a rater from a ``provider:model[:effort]`` spec (e.g.
    ``anthropic:claude-opus-4-8:low``, ``openai:gpt-5.4:medium``). The model is
    required and passed through as-is -- use whatever id your deployment serves.
    No effort part (or a trailing colon) sends no effort param."""
    provider, model, effort = parse_spec(spec)
    if provider == "anthropic":
        return AnthropicRater(model, effort=effort)
    return OpenAIRater(model, reasoning_effort=effort)
