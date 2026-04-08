"""
Unified LLM client for PaperMind.
Auto-selects provider based on environment variables.
Priority: GROQ_API_KEY > GEMINI_API_KEY > OPENROUTER_API_KEY > stub fallback.
"""
import json
import logging
import os

logger = logging.getLogger(__name__)


class LLMUnavailableError(Exception):
    """Raised when all LLM providers fail to respond."""
    pass


def _call_groq(prompt: str, system: str, max_tokens: int, temperature: float) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def _call_gemini(prompt: str, system: str, max_tokens: int, temperature: float) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system if system else None,
        generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
    )
    response = model.generate_content(prompt)
    return response.text


def _call_openrouter(prompt: str, system: str, max_tokens: int, temperature: float) -> str:
    import httpx
    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "HTTP-Referer": "https://papermind.local",
            "X-Title": "PaperMind",
        },
        json={
            "model": "mistralai/mistral-7b-instruct",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def call_llm(
    prompt: str,
    system: str = "You are a helpful research assistant.",
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> str:
    """
    Call the available LLM provider and return the response as a string.
    Tries providers in order: Groq → Gemini → OpenRouter.
    Raises LLMUnavailableError if all providers fail.
    """
    providers = []
    if os.environ.get("GROQ_API_KEY"):
        providers.append(("Groq", _call_groq))
    if os.environ.get("GEMINI_API_KEY"):
        providers.append(("Gemini", _call_gemini))
    if os.environ.get("OPENROUTER_API_KEY"):
        providers.append(("OpenRouter", _call_openrouter))

    for name, fn in providers:
        try:
            logger.debug(f"Calling LLM via {name}")
            return fn(prompt, system, max_tokens, temperature)
        except Exception as e:
            logger.warning(f"{name} call failed: {e}. Trying next provider.")

    logger.error("No LLM provider succeeded.")
    raise LLMUnavailableError(
        "All LLM providers failed. Set GROQ_API_KEY, GEMINI_API_KEY, or OPENROUTER_API_KEY in your environment."
    )


def call_llm_json(
    prompt: str,
    system: str = "You are a helpful research assistant.",
    max_tokens: int = 1024,
) -> dict:
    """
    Call LLM and parse the response as JSON.
    Strips markdown code fences before parsing.
    Returns empty dict on any failure.
    """
    raw = call_llm(prompt, system=system, max_tokens=max_tokens, temperature=0.1)
    try:
        # Strip markdown fences if present
        clean = raw.strip()
        if clean.startswith("```"):
            lines = clean.splitlines()
            # Remove first line (```json or ```) and last line (```)
            clean = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        return json.loads(clean)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}\nRaw response: {raw[:300]}")
        return {}
