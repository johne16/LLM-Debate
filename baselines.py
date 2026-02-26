"""
Baselines: Direct QA and Self-Consistency baselines (standalone script).

Baseline 1 - Direct QA: Single LLM answers with CoT prompting, no debate.
Baseline 2 - Self-Consistency: Sample N answers from single model, majority vote.
"""

import json
import os
import re
import anthropic
from dotenv import load_dotenv

from api_utils import call_llm
from data import fetch_strategy_qa
from logging_utils import save_baseline_log


PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def _load_prompt(filename: str) -> str:
    with open(os.path.join(PROMPTS_DIR, filename), "r", encoding="utf-8") as f:
        return f.read()


def _parse_answer(response_text: str) -> str:
    """Extract Yes/No answer from response."""
    match = re.search(r"ANSWER:\s*(Yes|No)", response_text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    text_lower = response_text.lower().strip()
    if text_lower.startswith("yes"):
        return "Yes"
    elif text_lower.startswith("no"):
        return "No"
    return "Unknown"


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    sample_count = config["self_consistency"]["sample_count"]
    if sample_count % 2 == 0:
        raise ValueError(f"self_consistency.sample_count must be odd, got {sample_count}")
    return config


def run_direct_qa(client: anthropic.Anthropic, question: str,
                  model: str, temperature: float, max_tokens: int) -> dict:
    """
    Baseline 1: Direct QA with CoT prompting.

    Returns:
        Dict with 'answer', 'reasoning', 'raw_response'.
    """
    template = _load_prompt("direct_qa.txt")
    prompt = template.format(question=question)
    response_text = call_llm(client, model, prompt, temperature, max_tokens)
    answer = _parse_answer(response_text)

    return {
        "answer": answer,
        "reasoning": response_text,
        "raw_response": response_text
    }


def run_self_consistency(client: anthropic.Anthropic, question: str,
                         model: str, temperature: float, max_tokens: int,
                         sample_count: int) -> dict:
    """
    Baseline 2: Self-Consistency via majority vote over N samples.

    Returns:
        Dict with 'answer', 'samples' (list), 'vote_counts'.
    """
    template = _load_prompt("self_consistency.txt")
    prompt = template.format(question=question)

    samples = []
    for i in range(sample_count):
        response_text = call_llm(client, model, prompt, temperature, max_tokens)
        answer = _parse_answer(response_text)
        samples.append({
            "answer": answer,
            "raw_response": response_text
        })

    # Majority vote
    yes_count = sum(1 for s in samples if s["answer"] == "Yes")
    no_count = sum(1 for s in samples if s["answer"] == "No")

    if yes_count > no_count:
        final_answer = "Yes"
    else:
        final_answer = "No"

    return {
        "answer": final_answer,
        "samples": samples,
        "vote_counts": {"Yes": yes_count, "No": no_count}
    }


def run_all_baselines(config_path: str = None) -> dict:
    """
    Run both baselines on StrategyQA data and save logs.

    Returns:
        Dict with 'direct_qa' and 'self_consistency' results and accuracies.
    """
    load_dotenv()
    config = load_config(config_path)
    client = anthropic.Anthropic()

    # Use debate sample_size for consistency
    sample_size = config["debate"]["sample_size"]
    print(f"Fetching {sample_size} StrategyQA questions...")
    questions = fetch_strategy_qa(sample_size)
    print(f"Fetched {len(questions)} questions.\n")

    # ---- Baseline 1: Direct QA ----
    dqa_cfg = config["direct_qa"]
    print("=" * 50)
    print("BASELINE 1: Direct QA")
    print("=" * 50)

    direct_qa_results = []
    for i, q in enumerate(questions):
        print(f"  [{i + 1}/{len(questions)}] {q['question']}")
        result = run_direct_qa(
            client, q["question"],
            dqa_cfg["model"], dqa_cfg["temperature"], dqa_cfg["max_tokens"]
        )
        ground_truth_label = "Yes" if q["answer"] else "No"
        result["ground_truth"] = q["answer"]
        result["ground_truth_label"] = ground_truth_label
        result["correct"] = result["answer"] == ground_truth_label
        result["question"] = q["question"]
        print(f"    Answer: {result['answer']} | GT: {ground_truth_label} | Correct: {result['correct']}")
        direct_qa_results.append(result)

    direct_qa_accuracy = sum(1 for r in direct_qa_results if r["correct"]) / len(direct_qa_results) if direct_qa_results else 0.0
    direct_qa_log = save_baseline_log("direct_qa", direct_qa_results, dqa_cfg)
    print(f"\n  Direct QA Accuracy: {direct_qa_accuracy:.2%}")
    print(f"  Log saved: {direct_qa_log}\n")

    # ---- Baseline 2: Self-Consistency ----
    sc_cfg = config["self_consistency"]
    print("=" * 50)
    print("BASELINE 2: Self-Consistency")
    print("=" * 50)

    self_consistency_results = []
    for i, q in enumerate(questions):
        print(f"  [{i + 1}/{len(questions)}] {q['question']}")
        result = run_self_consistency(
            client, q["question"],
            sc_cfg["model"], sc_cfg["temperature"], sc_cfg["max_tokens"],
            sc_cfg["sample_count"]
        )
        ground_truth_label = "Yes" if q["answer"] else "No"
        result["ground_truth"] = q["answer"]
        result["ground_truth_label"] = ground_truth_label
        result["correct"] = result["answer"] == ground_truth_label
        result["question"] = q["question"]
        print(f"    Answer: {result['answer']} (votes: {result['vote_counts']}) | "
              f"GT: {ground_truth_label} | Correct: {result['correct']}")
        self_consistency_results.append(result)

    self_consistency_accuracy = sum(1 for r in self_consistency_results if r["correct"]) / len(self_consistency_results) if self_consistency_results else 0.0
    self_consistency_log = save_baseline_log("self_consistency", self_consistency_results, sc_cfg)
    print(f"\n  Self-Consistency Accuracy: {self_consistency_accuracy:.2%}")
    print(f"  Log saved: {self_consistency_log}\n")

    # Summary
    print("=" * 50)
    print("BASELINE SUMMARY")
    print(f"  Direct QA:        {direct_qa_accuracy:.2%}")
    print(f"  Self-Consistency: {self_consistency_accuracy:.2%}")
    print("=" * 50)

    return {
        "direct_qa": {"results": direct_qa_results, "accuracy": direct_qa_accuracy, "log_path": direct_qa_log},
        "self_consistency": {"results": self_consistency_results, "accuracy": self_consistency_accuracy, "log_path": self_consistency_log}
    }


if __name__ == "__main__":
    run_all_baselines()
