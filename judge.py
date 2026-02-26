"""
Judge module: Evaluates a completed debate and renders a verdict.
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


def _parse_judge_response(response_text: str,
                          debater_a_answer: str = None,
                          debater_b_answer: str = None) -> dict:
    """
    Parse the judge's structured response into components.

    Args:
        response_text: Raw judge response.
        debater_a_answer: Debater A's final answer (for fallback recovery).
        debater_b_answer: Debater B's final answer (for fallback recovery).

    Returns:
        Dict with keys: cot_analysis, strongest_a, weakest_a, strongest_b,
        weakest_b, verdict, answer, confidence.
    """
    result = {
        "cot_analysis": "",
        "strongest_a": "",
        "weakest_a": "",
        "strongest_b": "",
        "weakest_b": "",
        "verdict": "",
        "answer": "Unknown",
        "confidence": 3
    }

    # Extract each field using regex
    patterns = {
        "cot_analysis": r"COT_ANALYSIS:\s*(.+?)(?=STRONGEST_A:|$)",
        "strongest_a": r"STRONGEST_A:\s*(.+?)(?=WEAKEST_A:|$)",
        "weakest_a": r"WEAKEST_A:\s*(.+?)(?=STRONGEST_B:|$)",
        "strongest_b": r"STRONGEST_B:\s*(.+?)(?=WEAKEST_B:|$)",
        "weakest_b": r"WEAKEST_B:\s*(.+?)(?=VERDICT:|$)",
        "verdict": r"VERDICT:\s*(.+?)(?=ANSWER:|$)",
        "answer": r"ANSWER:\s*(Yes|No)",
        "confidence": r"CONFIDENCE:\s*([1-5])",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if key == "confidence":
                result[key] = int(value)
            elif key == "answer":
                result[key] = value.capitalize()
            else:
                result[key] = value

    # Fallback: if answer not found, infer from verdict
    if result["answer"] == "Unknown" and result["verdict"]:
        verdict_lower = result["verdict"].lower()
        if "debater a" in verdict_lower:
            result["verdict"] = "Debater A"
            if debater_a_answer:
                result["answer"] = debater_a_answer
        elif "debater b" in verdict_lower:
            result["verdict"] = "Debater B"
            if debater_b_answer:
                result["answer"] = debater_b_answer

    return result


def evaluate_debate(client: anthropic.Anthropic, question: str,
                    transcript: str, model: str, temperature: float,
                    max_tokens: int,
                    debater_a_answer: str = None,
                    debater_b_answer: str = None) -> dict:
    """
    Judge evaluates the full debate transcript (Phase 3).

    Args:
        client: Anthropic client.
        question: The original StrategyQA question.
        transcript: Complete debate transcript.
        model: Judge model ID.
        temperature: Judge sampling temperature.
        max_tokens: Max tokens.
        debater_a_answer: Debater A's final answer (for fallback recovery).
        debater_b_answer: Debater B's final answer (for fallback recovery).

    Returns:
        Dict with parsed judge output: cot_analysis, strongest/weakest per side,
        verdict, answer (Yes/No), confidence (1-5), raw_response.
    """
    system_text = _load_prompt("judge_system.txt")
    user_content = f"Question: {question}\n\nFull Debate Transcript:\n{transcript}"

    response_text = call_llm(client, model, system=system_text,
                             messages=[{"role": "user", "content": user_content}],
                             temperature=temperature, max_tokens=max_tokens)

    parsed = _parse_judge_response(response_text,
                                   debater_a_answer=debater_a_answer,
                                   debater_b_answer=debater_b_answer)
    parsed["raw_response"] = response_text

    return parsed


def evaluate_debate_multi(client: anthropic.Anthropic, question: str,
                          transcript: str, debater_a_answer: str,
                          debater_b_answer: str, model: str,
                          temperature: float, max_tokens: int,
                          num_judges: int) -> dict:
    """
    Run multiple judges and aggregate by majority vote.

    Args:
        client: Anthropic client.
        question: The original question.
        transcript: Full debate transcript.
        debater_a_answer: Debater A's final answer.
        debater_b_answer: Debater B's final answer.
        model: Judge model ID.
        temperature: Judge temperature.
        max_tokens: Max tokens.
        num_judges: Number of judges (1 or 3).

    Returns:
        Dict with: individual_judgments (list), final_answer (str),
        final_verdict (str), avg_confidence (float).
    """
    judgments = []
    for i in range(num_judges):
        print(f"  Judge {i + 1}/{num_judges} evaluating...")
        judgment = evaluate_debate(client, question, transcript,
                                   model, temperature, max_tokens,
                                   debater_a_answer=debater_a_answer,
                                   debater_b_answer=debater_b_answer)
        judgments.append(judgment)

    # Majority vote on answer
    yes_count = sum(1 for j in judgments if j["answer"] == "Yes")
    no_count = sum(1 for j in judgments if j["answer"] == "No")

    if yes_count > no_count:
        final_answer = "Yes"
    elif no_count > yes_count:
        final_answer = "No"
    else:
        # Tie: use first judge's answer
        final_answer = judgments[0]["answer"]

    # Determine verdict based on final answer
    if debater_a_answer == debater_b_answer:
        # Both debaters converged on the same answer; no winner
        final_verdict = "Agreed"
    elif final_answer == debater_a_answer:
        final_verdict = "Debater A"
    elif final_answer == debater_b_answer:
        final_verdict = "Debater B"
    else:
        final_verdict = judgments[0].get("verdict", "Unknown")

    avg_confidence = sum(j["confidence"] for j in judgments) / len(judgments)

    return {
        "individual_judgments": judgments,
        "final_answer": final_answer,
        "final_verdict": final_verdict,
        "avg_confidence": avg_confidence
    }
