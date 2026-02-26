"""
Logging utilities: Save JSON transcripts of debate runs to the logs/ directory.
"""

import os
import json
from datetime import datetime


LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def ensure_logs_dir():
    """Create the logs directory if it doesn't exist."""
    os.makedirs(LOGS_DIR, exist_ok=True)


def save_debate_log(question: str, ground_truth,
                    initial_positions: dict, rounds: list,
                    judge_result: dict, config: dict,
                    consensus_skip: bool = False) -> str:
    """
    Save a complete debate transcript as a JSON log file.

    Args:
        question: The original question.
        ground_truth: True = Yes, False = No, None = unknown.
        initial_positions: Dict with 'debater_a' and 'debater_b' initial positions.
        rounds: List of round dicts, each with debater_a and debater_b arguments.
        judge_result: Judge evaluation result dict.
        config: The config dict used for this run.
        consensus_skip: Whether debate was skipped due to initial consensus.

    Returns:
        Path to the saved log file.
    """
    ensure_logs_dir()

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
    filename = f"single_debate_{timestamp}.json"
    filepath = os.path.join(LOGS_DIR, filename)

    if ground_truth is None:
        ground_truth_label = None
    else:
        ground_truth_label = "Yes" if ground_truth else "No"

    log_entry = {
        "timestamp": now.isoformat(),
        "question": question,
        "ground_truth": ground_truth,
        "ground_truth_label": ground_truth_label,
        "config": config,
        "consensus_skip": consensus_skip,
        "initial_positions": initial_positions,
        "rounds": rounds,
        "judge_result": judge_result,
        "final_answer": judge_result.get("final_answer", judge_result.get("answer", "Unknown")),
        "correct": _check_correct(judge_result, ground_truth)
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)

    return filepath


def _check_correct(judge_result: dict, ground_truth) -> bool | None:
    """Check if the judge's final answer matches ground truth.
    Returns None if ground_truth is None."""
    if ground_truth is None:
        return None
    final_answer = judge_result.get("final_answer", judge_result.get("answer", "Unknown"))
    ground_truth_label = "Yes" if ground_truth else "No"
    return final_answer == ground_truth_label


def save_debate_batch_log(results: list, config: dict) -> str:
    """
    Save batch debate results as a single log file.

    Args:
        results: List of result dicts from run_single_debate.
        config: Config dict used.

    Returns:
        Path to saved log file.
    """
    ensure_logs_dir()

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
    filename = f"debate_{timestamp}.json"
    filepath = os.path.join(LOGS_DIR, filename)

    scored = [r for r in results if r.get("correct") is not None]
    log_entry = {
        "timestamp": now.isoformat(),
        "config": config,
        "results": results,
        "accuracy": sum(1 for r in scored if r["correct"]) / len(scored) if scored else 0.0
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)

    return filepath


def save_baseline_log(method: str, results: list, config: dict) -> str:
    """
    Save baseline experiment results.

    Args:
        method: 'direct_qa' or 'self_consistency'.
        results: List of result dicts per question.
        config: Config dict used.

    Returns:
        Path to saved log file.
    """
    ensure_logs_dir()

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
    filename = f"baseline_{method}_{timestamp}.json"
    filepath = os.path.join(LOGS_DIR, filename)

    log_entry = {
        "timestamp": now.isoformat(),
        "method": method,
        "config": config,
        "results": results,
        "accuracy": sum(1 for r in results if r["correct"]) / len(results) if results else 0.0
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)

    return filepath
