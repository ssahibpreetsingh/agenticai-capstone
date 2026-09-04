"""
config.py — Provider-agnostic LLM access layer.

This is the ONLY file that should know which LLM provider you're using.
Every other script calls `chat(messages)` and doesn't care whether that's
Azure OpenAI, OpenAI, Anthropic, an OpenAI-compatible endpoint, or Ollama.

You should NOT need to edit this file — just fill in `.env` correctly.
Run this file directly to sanity-check your setup:

    python starter_code/config.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
SCENARIO = os.getenv("SCENARIO", "banking").lower()  # "banking" or "healthcare"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", SCENARIO)


def chat(messages: list[dict], temperature: float = 0.2, max_tokens: int = 600) -> str:
    """
    Universal chat call. `messages` is a list of {"role": "system"/"user"/"assistant",
    "content": "..."} dicts, same shape as OpenAI's chat format.

    Returns the assistant's text response as a plain string.
    """
    if PROVIDER == "openrouter":
        return _chat_openRouter(messages,temperature, max_tokens)
    if PROVIDER == "azure":
        return _chat_azure(messages, temperature, max_tokens)
    elif PROVIDER == "openai":
        return _chat_openai(messages, temperature, max_tokens)
    elif PROVIDER == "anthropic":
        return _chat_anthropic(messages, temperature, max_tokens)
    elif PROVIDER == "openai_compatible":
        return _chat_openai_compatible(messages, temperature, max_tokens)
    elif PROVIDER == "ollama":
        return _chat_ollama(messages, temperature, max_tokens)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{PROVIDER}'. Must be one of: "
            "azure, openai, anthropic, openai_compatible, ollama"
        )


# ---------------------------------------------------------------------------
# Provider implementations — you shouldn't need to touch these, but they're
# short and readable if you want to see exactly what's happening.
# ---------------------------------------------------------------------------


def _chat_openRouter(messages,temperature, max_tokens):
    from openrouter import OpenRouter
    with OpenRouter(
      api_key=os.getenv("OPENROUTER_API_KEY", ""),
    ) as client:
            response = client.chat.send(model="nvidia/nemotron-3-super-120b-a12b:free",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens)
    
    return response.choices[0].message.content

def _chat_openai(messages, temperature, max_tokens):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def _chat_azure(messages, temperature, max_tokens):
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
    )
    resp = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def _chat_anthropic(messages, temperature, max_tokens):
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    system = ""
    user_messages = []
    for m in messages:
        if m["role"] == "system":
            system += m["content"] + "\n"
        else:
            user_messages.append(m)
    resp = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        system=system.strip(),
        messages=user_messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.content[0].text


def _chat_openai_compatible(messages, temperature, max_tokens):
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY"),
        base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL"),
    )
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_COMPATIBLE_MODEL"),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def _chat_ollama(messages, temperature, max_tokens):
    import requests
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    resp = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# ---------------------------------------------------------------------------
# Shared embedding model (always local/free, regardless of LLM provider)
# ---------------------------------------------------------------------------

_embedding_model = None


def get_embedding_model():
    """Lazily loads and caches the sentence-transformers embedding model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


# ---------------------------------------------------------------------------
# Sanity check — run this file directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[OK] Provider detected: {PROVIDER}")
    print(f"[OK] Scenario: {SCENARIO}  (data dir: {DATA_DIR})")

    try:
        reply = chat([{"role": "user", "content": "Reply with exactly: Hello from capstone setup."}])
        print(f"[OK] Test call succeeded: \"{reply.strip()}\"")
    except Exception as e:
        print(f"[FAIL] LLM test call failed: {e}")
        print("       Check your .env file against SETUP.md")

    try:
        model = get_embedding_model()
        vec = model.encode("test sentence")
        print(f"[OK] Embedding model loaded: {EMBEDDING_MODEL_NAME} (dim={len(vec)})")
    except Exception as e:
        print(f"[FAIL] Embedding model failed to load: {e}")
