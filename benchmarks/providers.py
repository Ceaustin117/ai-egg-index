"""Multi-provider LLM dispatch for benchmark evaluators.

Each provider has a thin wrapper that takes the same (model, prompt, max_tokens, temperature)
signature and returns the response text. Errors are returned as 'ERROR: ...' strings so they
flow through the existing scoring pipeline (the judge will score them as failures).
"""

import os
import random
import re
import time
from typing import Optional

SUPPORTED_PROVIDERS = ("groq", "google", "cohere", "huggingface")

# Max retries on transient rate-limit / quota (HTTP 429) errors, per provider.
# Google's free tier hits short-window quota easily; without retries a single 429
# turns every answer into an "ERROR: ..." string that the judge scores as 0, zeroing
# the model's whole scorecard for that run. Other providers haven't needed it (0).
# Keep these bounded: retries × backoff happens per failing question, so generous
# values can blow the CI job's time budget (a 5×60s version hit the 60-min cap).
_MAX_RETRIES = {
    "google": 3,
}
_MAX_BACKOFF_S = 20.0

# Circuit breaker: after this many consecutive rate-limit failures, a provider is
# treated as down for the rest of the run (sustained quota exhaustion) and further
# calls skip the retry budget — so one throttled provider can't run out the clock.
# Any success resets the counter. Per-process state (resets each run).
_CIRCUIT_THRESHOLD = 3
_consecutive_rate_limits: dict[str, int] = {}

# Minimum seconds between consecutive calls per provider. Used to stay under
# free-tier RPM limits. Only providers that have hit limits are throttled.
# Google Gemini 2.5 Flash free tier = 10 RPM; 7s interval => ~8.5 RPM with headroom.
_MIN_INTERVAL_S = {
    "google": 7.0,
}
_LAST_CALL_TS: dict[str, float] = {}


def _throttle(provider: str) -> None:
    interval = _MIN_INTERVAL_S.get(provider, 0)
    if interval <= 0:
        return
    last = _LAST_CALL_TS.get(provider, 0.0)
    wait = interval - (time.time() - last)
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL_TS[provider] = time.time()

API_KEY_ENV_VARS = {
    "groq": "GROQ_API_KEY",
    "google": "GOOGLE_API_KEY",
    "cohere": "COHERE_API_KEY",
    "huggingface": "HF_TOKEN",
}


class ProviderError(Exception):
    """Raised for missing API keys or unknown providers (config issues, not API failures)."""


def _is_rate_limit(exc: Exception) -> bool:
    """True if an exception looks like a transient rate-limit / quota (HTTP 429)."""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return (
        "resourceexhausted" in name
        or "ratelimit" in name
        or "429" in text
        or "quota" in text
        or "rate limit" in text
    )


def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
    """Honor the server's suggested retry delay (e.g. 'Please retry in 4.7s') when
    present; otherwise exponential backoff with jitter, capped at _MAX_BACKOFF_S."""
    match = re.search(r"retry in ([\d.]+)\s*s", str(exc), re.IGNORECASE)
    if match:
        return min(float(match.group(1)) + random.uniform(0, 1), _MAX_BACKOFF_S)
    return min(2.0 ** attempt + random.uniform(0, 1), _MAX_BACKOFF_S)


def provider_api_key(provider: str) -> Optional[str]:
    env_var = API_KEY_ENV_VARS.get(provider)
    return os.environ.get(env_var) if env_var else None


def call_llm(
    provider: str,
    model: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    if provider not in SUPPORTED_PROVIDERS:
        raise ProviderError(f"Unknown provider '{provider}'. Supported: {SUPPORTED_PROVIDERS}")
    if not provider_api_key(provider):
        raise ProviderError(
            f"Missing {API_KEY_ENV_VARS[provider]} for provider '{provider}'"
        )

    dispatcher = {
        "groq": _call_groq,
        "google": _call_google,
        "cohere": _call_cohere,
        "huggingface": _call_huggingface,
    }[provider]

    # If the provider has already rate-limited repeatedly this run, the breaker is open:
    # don't spend retry budget (assume sustained quota exhaustion). One success closes it.
    breaker_open = _consecutive_rate_limits.get(provider, 0) >= _CIRCUIT_THRESHOLD
    max_retries = 0 if breaker_open else _MAX_RETRIES.get(provider, 0)
    if breaker_open:
        print(f"  [{provider}] circuit open (sustained rate limits); not retrying", flush=True)

    # Throttle once for normal call spacing; retry backoff handles spacing thereafter.
    _throttle(provider)
    for attempt in range(max_retries + 1):
        try:
            result = dispatcher(model, prompt, max_tokens, temperature)
            _consecutive_rate_limits[provider] = 0
            return result
        except Exception as e:
            if _is_rate_limit(e):
                if attempt < max_retries:
                    delay = _retry_delay_seconds(e, attempt)
                    print(
                        f"  [{provider}] rate-limited (attempt {attempt + 1}/{max_retries + 1}); "
                        f"retrying in {delay:.1f}s",
                        flush=True,
                    )
                    time.sleep(delay)
                    continue
                _consecutive_rate_limits[provider] = _consecutive_rate_limits.get(provider, 0) + 1
            return f"ERROR: {type(e).__name__}: {e}"


def _call_groq(model, prompt, max_tokens, temperature):
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def _call_google(model, prompt, max_tokens, temperature):
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    m = genai.GenerativeModel(model)
    resp = m.generate_content(
        prompt,
        generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
    )
    return resp.text


def _call_cohere(model, prompt, max_tokens, temperature):
    import cohere

    client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
    resp = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.message.content[0].text


def _call_huggingface(model, prompt, max_tokens, temperature):
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=os.environ["HF_TOKEN"])
    resp = client.chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content
