"""Multi-provider LLM dispatch for benchmark evaluators.

Each provider has a thin wrapper that takes the same (model, prompt, max_tokens, temperature)
signature and returns the response text. Errors are returned as 'ERROR: ...' strings so they
flow through the existing scoring pipeline (the judge will score them as failures).
"""

import os
import time
from typing import Optional

SUPPORTED_PROVIDERS = ("groq", "together", "google", "cohere", "huggingface")

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
    "together": "TOGETHER_API_KEY",
    "google": "GOOGLE_API_KEY",
    "cohere": "COHERE_API_KEY",
    "huggingface": "HF_TOKEN",
}


class ProviderError(Exception):
    """Raised for missing API keys or unknown providers (config issues, not API failures)."""


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
        "together": _call_together,
        "google": _call_google,
        "cohere": _call_cohere,
        "huggingface": _call_huggingface,
    }[provider]

    _throttle(provider)
    try:
        return dispatcher(model, prompt, max_tokens, temperature)
    except Exception as e:
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


def _call_together(model, prompt, max_tokens, temperature):
    from together import Together

    client = Together(api_key=os.environ["TOGETHER_API_KEY"])
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
