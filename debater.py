"""
Debater module: Handles initial position generation and debate rounds for both debaters.
"""

import os
import re
import anthropic
from api_utils import call_llm


PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def _load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    with open(os.path.join(PROMPTS_DIR, filename), "r", encoding="utf-8") as f:
        return f.read()


def _parse_answer(response_text: str) -> str:
    """
    Extract the answer (Yes/No) from a debater's response.
    Returns 'Yes', 'No', or 'Unknown' if parsing fails.
    """
    match = re.search(r"ANSWER:\s*(Yes|No)", response_text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    # Fallback: look for yes/no at the start of the response
    text_lower = response_text.lower().strip()
    if text_lower.startswith("yes"):
        return "Yes"
    elif text_lower.startswith("no"):
        return "No"
    return "Unknown"


def generate_initial_position(client: anthropic.Anthropic, question: str,
                              role: str, model: str, temperature: float,
                              max_tokens: int) -> dict:
    """
    Generate an initial position for a debater (Phase 1).

    Args:
        client: Anthropic client.
        question: The StrategyQA question.
        role: 'A' or 'B'.
        model: Model ID.
        temperature: Sampling temperature.
        max_tokens: Max tokens.

    Returns:
        Dict with 'answer' (str), 'reasoning' (str), 'raw_response' (str).
    """
    if role == "A":
        system_text = _load_prompt("debater_a_initial_system.txt")
    else:
        system_text = _load_prompt("debater_b_initial_system.txt")

    messages = [{"role": "user", "content": f"Question: {question}"}]
    response_text = call_llm(client, model, system=system_text,
                             messages=messages, temperature=temperature,
                             max_tokens=max_tokens)

    answer = _parse_answer(response_text)

    return {
        "answer": answer,
        "reasoning": response_text,
        "raw_response": response_text
    }


def generate_debate_argument(client: anthropic.Anthropic, question: str,
                             role: str, position: str,
                             initial_positions: dict, rounds: list,
                             opponent_current_argument: str,
                             model: str, temperature: float,
                             max_tokens: int) -> dict:
    """
    Generate a debate argument for a given round (Phase 2).

    Uses multi-turn conversation structure: previous debate exchanges are
    represented as alternating user/assistant messages.

    Args:
        client: Anthropic client.
        question: The StrategyQA question.
        role: 'A' or 'B'.
        position: The debater's current answer ('Yes' or 'No').
        initial_positions: Dict with 'debater_a' and 'debater_b' initial data.
        rounds: List of completed round dicts (each has 'debater_a' and 'debater_b').
        opponent_current_argument: The opponent's argument for the current round
            (for Debater B, this is A's current-round argument; for Debater A, empty string).
        model: Model ID.
        temperature: Sampling temperature.
        max_tokens: Max tokens.

    Returns:
        Dict with 'answer' (str), 'argument' (str), 'raw_response' (str).
    """
    if role == "A":
        system_template = _load_prompt("debater_a_round_system.txt")
        self_key = "debater_a"
        opponent_key = "debater_b"
    else:
        system_template = _load_prompt("debater_b_round_system.txt")
        self_key = "debater_b"
        opponent_key = "debater_a"

    system_text = system_template.format(position=position)

    # Build multi-turn messages from debate history
    messages = []

    # First user message: the question + initial positions
    init_text = (
        f"Question: {question}\n\n"
        f"=== INITIAL POSITIONS ===\n"
        f"Debater A (Initial Position):\n"
        f"Answer: {initial_positions['debater_a']['answer']}\n"
        f"{initial_positions['debater_a']['reasoning']}\n\n"
        f"Debater B (Initial Position):\n"
        f"Answer: {initial_positions['debater_b']['answer']}\n"
        f"{initial_positions['debater_b']['reasoning']}"
    )
    messages.append({"role": "user", "content": init_text})

    # Insert debater's own initial position as an assistant message
    # to maintain alternating user/assistant roles before the next user message
    self_initial = initial_positions[self_key]
    messages.append({"role": "assistant", "content": self_initial['reasoning']})

    # For each completed round, represent opponent's argument as user message
    # and this debater's response as assistant message
    for round_data in rounds:
        opponent_argument = round_data[opponent_key]
        self_argument = round_data[self_key]

        opponent_text = (
            f"{opponent_key.replace('_', ' ').title()} argument:\n"
            f"Answer: {opponent_argument['answer']}\n"
            f"{opponent_argument['argument']}"
        )
        messages.append({"role": "user", "content": opponent_text})

        self_text = (
            f"Answer: {self_argument['answer']}\n"
            f"{self_argument['argument']}"
        )
        messages.append({"role": "assistant", "content": self_text})

    # The current round's opponent argument as the final user message
    if opponent_current_argument:
        opponent_label = opponent_key.replace("_", " ").title()
        current_text = (
            f"{opponent_label} argument:\n"
            f"{opponent_current_argument}"
        )
        messages.append({"role": "user", "content": current_text})
    else:
        # Debater A goes first in a round; prompt them to respond
        messages.append({"role": "user", "content": "It is your turn to present your argument for this round."})

    response_text = call_llm(client, model, system=system_text,
                             messages=messages, temperature=temperature,
                             max_tokens=max_tokens)

    answer = _parse_answer(response_text)

    return {
        "answer": answer,
        "argument": response_text,
        "raw_response": response_text
    }
