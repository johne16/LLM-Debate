"""
Shared API utility: Anthropic LLM call wrapper with exponential backoff retry logic.
"""

import time
import anthropic


def call_llm(client: anthropic.Anthropic, model: str, prompt: str = None,
             temperature: float = 0.7, max_tokens: int = 500,
             max_retries: int = 5, base_delay: float = 2.0,
             system: str = None, messages: list = None) -> str:
    """
    Call the Anthropic API with exponential backoff retry on rate limit
    and transient errors.

    Supports two calling conventions:
      1. Legacy: pass `prompt` as a single user message string.
      2. Multi-turn: pass `messages` (list of role/content dicts) and
         optionally `system` (string for system prompt).

    If both `prompt` and `messages` are provided, `messages` takes precedence.

    Args:
        client: Anthropic client instance.
        model: Model ID string.
        prompt: User message content (legacy shortcut).
        temperature: Sampling temperature.
        max_tokens: Max tokens in response.
        max_retries: Number of retry attempts.
        base_delay: Base delay in seconds for exponential backoff.
        system: Optional system prompt string.
        messages: Optional list of message dicts with 'role' and 'content'.

    Returns:
        The text content of the model's response.
    """
    # Build the messages list
    if messages is not None:
        msg_list = messages
    elif prompt is not None:
        msg_list = [{"role": "user", "content": prompt}]
    else:
        raise ValueError("Either 'prompt' or 'messages' must be provided.")

    # Build API call kwargs
    api_kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": msg_list,
    }
    if system:
        api_kwargs["system"] = system

    for attempt in range(max_retries):
        try:
            response = client.messages.create(**api_kwargs)
            return response.content[0].text

        except anthropic.RateLimitError as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"  Rate limited (attempt {attempt + 1}/{max_retries}). "
                      f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise e

        except anthropic.APIStatusError as e:
            # Retry on 5xx server errors, raise on 4xx client errors (except rate limit)
            if e.status_code >= 500 and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"  Server error {e.status_code} (attempt {attempt + 1}/{max_retries}). "
                      f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise e

        except anthropic.APIConnectionError as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"  Connection error (attempt {attempt + 1}/{max_retries}). "
                      f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise e
