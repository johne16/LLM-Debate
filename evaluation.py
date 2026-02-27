"""
Evaluation module: Compare debate pipeline vs baselines.
Computes accuracy, statistical significance, and generates summary tables/figures.
"""

import csv
import json
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy import stats


LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def load_latest_log(prefix: str) -> dict | None:
    """Load the most recent log file matching a prefix from the logs directory."""
    pattern = os.path.join(LOGS_DIR, f"{prefix}*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    if not files:
        return None
    with open(files[0], "r", encoding="utf-8") as f:
        return json.load(f)


def compute_accuracy(results: list[dict]) -> float:
    """Compute accuracy from a list of result dicts with 'correct' key.
    Results where correct is None (e.g. custom questions) are excluded."""
    if not results:
        return 0.0
    scored = [r for r in results if r.get("correct") is not None]
    if not scored:
        return 0.0
    return sum(1 for r in scored if r["correct"]) / len(scored)


def mcnemar_test(results_a: list[bool], results_b: list[bool]) -> dict:
    """
    Perform McNemar's test for paired binary outcomes.

    Args:
        results_a: List of bool (correct/incorrect) for method A.
        results_b: List of bool (correct/incorrect) for method B.

    Returns:
        Dict with 'statistic', 'p_value', 'significant' (at alpha=0.05).
    """
    assert len(results_a) == len(results_b), "Result lists must be same length"

    # Build contingency table
    a_right_b_wrong = sum(1 for a, b in zip(results_a, results_b) if a and not b)
    a_wrong_b_right = sum(1 for a, b in zip(results_a, results_b) if not a and b)

    if a_right_b_wrong + a_wrong_b_right == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    # McNemar's test with continuity correction
    statistic = (abs(a_right_b_wrong - a_wrong_b_right) - 1) ** 2 / (a_right_b_wrong + a_wrong_b_right)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < 0.05
    }


def evaluate_all() -> dict:
    """
    Run full evaluation comparing debate pipeline vs baselines.
    Loads the most recent log file for each method and extracts
    both results and config parameters.

    Returns:
        Evaluation summary dict with accuracy, counts, and config for each method.
    """
    debate_log = load_latest_log("debate_")
    if debate_log is None:
        print("No debate logs found.")
        debate_log = {}

    direct_qa_log = load_latest_log("baseline_direct_qa_")
    if direct_qa_log is None:
        print("No Direct QA baseline logs found.")
        direct_qa_log = {}

    self_consistency_log = load_latest_log("baseline_self_consistency_")
    if self_consistency_log is None:
        print("No Self-Consistency baseline logs found.")
        self_consistency_log = {}

    debate_results = debate_log.get("results", [])
    direct_qa_results = direct_qa_log.get("results", [])
    self_consistency_results = self_consistency_log.get("results", [])

    # Compute accuracies
    debate_accuracy = compute_accuracy(debate_results)
    direct_qa_accuracy = compute_accuracy(direct_qa_results)
    self_consistency_accuracy = compute_accuracy(self_consistency_results)

    summary = {
        "debate": {
            "accuracy": debate_accuracy,
            "total": len(debate_results),
            "correct": sum(1 for r in debate_results if r.get("correct") is True),
            "config": debate_log.get("config", {})
        },
        "direct_qa": {
            "accuracy": direct_qa_accuracy,
            "total": len(direct_qa_results),
            "correct": sum(1 for r in direct_qa_results if r.get("correct") is True),
            "config": direct_qa_log.get("config", {})
        },
        "self_consistency": {
            "accuracy": self_consistency_accuracy,
            "total": len(self_consistency_results),
            "correct": sum(1 for r in self_consistency_results if r.get("correct") is True),
            "config": self_consistency_log.get("config", {})
        }
    }

    # Consensus rate: % of debate questions where debaters agreed at initialization
    scored_debate = [r for r in debate_results if r.get("correct") is not None]
    if scored_debate:
        consensus_count = sum(1 for r in scored_debate if r.get("consensus_skip"))
        summary["consensus_rate"] = consensus_count / len(scored_debate)
        summary["consensus_count"] = consensus_count
    else:
        summary["consensus_rate"] = 0.0
        summary["consensus_count"] = 0

    # Average judge confidence on correct vs incorrect questions
    correct_confidences = []
    incorrect_confidences = []
    for r in scored_debate:
        judge_result = r.get("judge_result", {})
        confidence = judge_result.get("avg_confidence")
        if confidence is not None:
            if r["correct"]:
                correct_confidences.append(confidence)
            else:
                incorrect_confidences.append(confidence)
    summary["avg_confidence_correct"] = (
        sum(correct_confidences) / len(correct_confidences) if correct_confidences else None
    )
    summary["avg_confidence_incorrect"] = (
        sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else None
    )

    # Statistical significance tests (if same number of scored questions)
    scored_dqa = [r for r in direct_qa_results if r.get("correct") is not None]
    scored_sc = [r for r in self_consistency_results if r.get("correct") is not None]

    if len(scored_debate) == len(scored_dqa) and len(scored_debate) > 0:
        debate_correct = [bool(r["correct"]) for r in scored_debate]
        direct_qa_correct = [bool(r["correct"]) for r in scored_dqa]
        summary["mcnemar_debate_vs_dqa"] = mcnemar_test(debate_correct, direct_qa_correct)

    if len(scored_debate) == len(scored_sc) and len(scored_debate) > 0:
        debate_correct = [bool(r["correct"]) for r in scored_debate]
        self_consistency_correct = [bool(r["correct"]) for r in scored_sc]
        summary["mcnemar_debate_vs_sc"] = mcnemar_test(debate_correct, self_consistency_correct)

    return summary


def save_evaluation_csv(summary: dict, output_path: str = None):
    """Save evaluation summary to CSV."""
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   f"results_summary_{TIMESTAMP}.csv")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Correct", "Total", "Accuracy"])
        for method_key, label in [("debate", "Debate Pipeline"),
                                   ("direct_qa", "Direct QA"),
                                   ("self_consistency", "Self-Consistency")]:
            data = summary.get(method_key, {})
            writer.writerow([label, data.get("correct", 0), data.get("total", 0),
                             f"{data.get('accuracy', 0.0):.4f}"])

        writer.writerow([])
        writer.writerow(["Metric", "Value"])
        if "consensus_rate" in summary:
            writer.writerow(["Consensus Rate", f"{summary['consensus_rate']:.4f}"])
            writer.writerow(["Consensus Count", summary.get("consensus_count", 0)])
        if summary.get("avg_confidence_correct") is not None:
            writer.writerow(["Avg Confidence (Correct)", f"{summary['avg_confidence_correct']:.2f}"])
        if summary.get("avg_confidence_incorrect") is not None:
            writer.writerow(["Avg Confidence (Incorrect)", f"{summary['avg_confidence_incorrect']:.2f}"])

        for test_key, label in [("mcnemar_debate_vs_dqa", "Debate vs Direct QA"),
                                 ("mcnemar_debate_vs_sc", "Debate vs Self-Consistency")]:
            if test_key in summary:
                test = summary[test_key]
                writer.writerow([])
                writer.writerow([f"McNemar ({label})", ""])
                writer.writerow(["Statistic", f"{test['statistic']:.4f}"])
                writer.writerow(["p-value", f"{test['p_value']:.4f}"])
                writer.writerow(["Significant (alpha=0.05)", "YES" if test["significant"] else "NO"])

        writer.writerow([])
        writer.writerow(["Configuration"])
        writer.writerow(["Method", "Parameter", "Value"])
        for method_key, label in [("debate", "Debate Pipeline"),
                                   ("direct_qa", "Direct QA"),
                                   ("self_consistency", "Self-Consistency")]:
            config = summary.get(method_key, {}).get("config", {})
            for param, value in config.items():
                writer.writerow([label, param, value])

    print(f"Results saved to: {output_path}")


def generate_results_figure(summary: dict, output_path: str = None):
    """
    Generate a bar chart comparing method accuracies.
    Saves to output_path.
    """
    methods = []
    accuracies = []
    for method_key, label in [("debate", "Debate\nPipeline"),
                               ("direct_qa", "Direct QA"),
                               ("self_consistency", "Self-\nConsistency")]:
        data = summary.get(method_key, {})
        methods.append(label)
        accuracies.append(data.get("accuracy", 0.0) * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, accuracies, color=["#2196F3", "#FF9800", "#4CAF50"],
                  edgecolor="black", linewidth=0.8)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("StrategyQA Accuracy: Debate Pipeline vs Baselines", fontsize=14)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   f"results_comparison_{TIMESTAMP}.png")
    plt.savefig(output_path, dpi=150)
    print(f"Figure saved to: {output_path}")
    plt.close()


def generate_heatmap(output_path: str = None):
    """
    Generate a per-question heatmap showing correct/incorrect across all three methods.
    Loads the most recent log file for each method.
    Questions on one axis, methods on the other.
    Saves to output_path or a default file.
    """

    debate_log = load_latest_log("debate_")
    debate_results = debate_log.get("results", []) if debate_log else []

    direct_qa_log = load_latest_log("baseline_direct_qa_")
    direct_qa_results = direct_qa_log.get("results", []) if direct_qa_log else []

    self_consistency_log = load_latest_log("baseline_self_consistency_")
    self_consistency_results = self_consistency_log.get("results", []) if self_consistency_log else []

    # Filter to scored only
    debate_scored = [r for r in debate_results if r.get("correct") is not None]
    direct_qa_scored = [r for r in direct_qa_results if r.get("correct") is not None]
    self_consistency_scored = [r for r in self_consistency_results if r.get("correct") is not None]

    # Use the minimum length across methods so rows align
    num_questions = min(len(debate_scored), len(direct_qa_scored), len(self_consistency_scored))
    if num_questions == 0:
        print("Not enough matched results across methods for heatmap.")
        return

    # Build data matrix: rows = questions, cols = methods
    # 1 = correct, 0 = incorrect
    data = np.zeros((num_questions, 3))
    for i in range(num_questions):
        data[i, 0] = 1 if debate_scored[i]["correct"] else 0
        data[i, 1] = 1 if direct_qa_scored[i]["correct"] else 0
        data[i, 2] = 1 if self_consistency_scored[i]["correct"] else 0

    fig, ax = plt.subplots(figsize=(5, max(4, num_questions * 0.35)))

    cmap = mcolors.ListedColormap(["#EF5350", "#66BB6A"])
    ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    # Labels
    q_labels = [f"Q{i+1}" for i in range(num_questions)]
    method_labels = ["Debate", "Direct QA", "Self-Consistency"]

    ax.set_xticks(range(3))
    ax.set_xticklabels(method_labels, fontsize=10)
    ax.set_yticks(range(num_questions))
    ax.set_yticklabels(q_labels, fontsize=8)

    ax.set_xlabel("Method")
    ax.set_ylabel("Question")
    ax.set_title("Per-Question Correctness Heatmap")

    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   f"results_heatmap_{TIMESTAMP}.png")
    plt.savefig(output_path, dpi=150)
    print(f"Heatmap saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    summary = evaluate_all()
    save_evaluation_csv(summary)
    generate_results_figure(summary)
    generate_heatmap()
