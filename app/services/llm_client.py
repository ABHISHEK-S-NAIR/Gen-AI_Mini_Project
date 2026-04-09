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


def _call_groq(
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
    messages: list[dict] | None = None,
) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    payload_messages = messages if messages is not None else [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=payload_messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def _call_gemini(
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
    messages: list[dict] | None = None,
) -> str:
    import google.generativeai as genai

    system_instruction = system
    content = prompt
    if messages is not None:
        system_messages = [m.get("content", "") for m in messages if m.get("role") == "system"]
        if system_messages:
            system_instruction = "\n\n".join(system_messages)

        gemini_messages: list[dict] = []
        for msg in messages:
            role = msg.get("role")
            if role == "system":
                continue
            gemini_role = "model" if role == "assistant" else "user"
            gemini_messages.append({"role": gemini_role, "parts": [msg.get("content", "")]})
        content = gemini_messages or prompt

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system_instruction if system_instruction else None,
        generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
    )
    response = model.generate_content(content)
    return response.text


def _call_openrouter(
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
    messages: list[dict] | None = None,
) -> str:
    import httpx

    payload_messages = messages if messages is not None else [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "HTTP-Referer": "https://papermind.local",
            "X-Title": "PaperMind",
        },
        json={
            "model": "mistralai/mistral-7b-instruct",
            "messages": payload_messages,
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
    messages: list[dict] | None = None,
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
            return fn(prompt, system, max_tokens, temperature, messages=messages)
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
