"""
Evaluation module: Compare debate pipeline vs baselines.
Computes accuracy, statistical significance, and generates summary tables/figures.
"""

import json
import os
import glob
from scipy import stats


LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


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


def evaluate_all(debate_results: list[dict] = None,
                 direct_qa_results: list[dict] = None,
                 self_consistency_results: list[dict] = None) -> dict:
    """
    Run full evaluation comparing debate pipeline vs baselines.

    Args:
        debate_results: List of debate result dicts (from orchestrator).
                       If None, loads from logs.
        direct_qa_results: Direct QA results. If None, loads from logs.
        self_consistency_results: Self-consistency results. If None, loads from logs.

    Returns:
        Evaluation summary dict.
    """
    # Load from logs if not provided
    if debate_results is None:
        debate_log = load_latest_log("debate_")
        if debate_log:
            debate_results = debate_log.get("results", [])
        else:
            print("No debate logs found.")
            debate_results = []

    if direct_qa_results is None:
        direct_qa_log = load_latest_log("baseline_direct_qa_")
        if direct_qa_log:
            direct_qa_results = direct_qa_log.get("results", [])
        else:
            print("No Direct QA baseline logs found.")
            direct_qa_results = []

    if self_consistency_results is None:
        self_consistency_log = load_latest_log("baseline_self_consistency_")
        if self_consistency_log:
            self_consistency_results = self_consistency_log.get("results", [])
        else:
            print("No Self-Consistency baseline logs found.")
            self_consistency_results = []

    # Compute accuracies
    debate_accuracy = compute_accuracy(debate_results)
    direct_qa_accuracy = compute_accuracy(direct_qa_results)
    self_consistency_accuracy = compute_accuracy(self_consistency_results)

    summary = {
        "debate": {
            "accuracy": debate_accuracy,
            "total": len(debate_results),
            "correct": sum(1 for r in debate_results if r.get("correct") is True)
        },
        "direct_qa": {
            "accuracy": direct_qa_accuracy,
            "total": len(direct_qa_results),
            "correct": sum(1 for r in direct_qa_results if r.get("correct") is True)
        },
        "self_consistency": {
            "accuracy": self_consistency_accuracy,
            "total": len(self_consistency_results),
            "correct": sum(1 for r in self_consistency_results if r.get("correct") is True)
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


def print_evaluation_table(summary: dict):
    """Print a formatted evaluation summary table."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    header = f"{'Method':<25} {'Correct':>8} {'Total':>8} {'Accuracy':>10}"
    print(header)
    print("-" * 60)

    for method_key, label in [("debate", "Debate Pipeline"),
                               ("direct_qa", "Direct QA"),
                               ("self_consistency", "Self-Consistency")]:
        data = summary.get(method_key, {})
        correct = data.get("correct", 0)
        total = data.get("total", 0)
        acc = data.get("accuracy", 0.0)
        print(f"{label:<25} {correct:>8} {total:>8} {acc:>9.2%}")

    print("-" * 60)

    # Consensus rate
    if "consensus_rate" in summary:
        rate = summary["consensus_rate"]
        count = summary.get("consensus_count", 0)
        total = summary.get("debate", {}).get("total", 0)
        print(f"\nConsensus rate: {rate:.2%} ({count}/{total} questions skipped debate)")

    # Judge confidence on correct vs incorrect
    conf_correct = summary.get("avg_confidence_correct")
    conf_incorrect = summary.get("avg_confidence_incorrect")
    if conf_correct is not None or conf_incorrect is not None:
        print(f"\nAvg judge confidence (1-5):")
        if conf_correct is not None:
            print(f"  On correct questions:   {conf_correct:.2f}")
        else:
            print(f"  On correct questions:   N/A")
        if conf_incorrect is not None:
            print(f"  On incorrect questions: {conf_incorrect:.2f}")
        else:
            print(f"  On incorrect questions: N/A")

    # Statistical significance
    for test_key, label in [("mcnemar_debate_vs_dqa", "Debate vs Direct QA"),
                             ("mcnemar_debate_vs_sc", "Debate vs Self-Consistency")]:
        if test_key in summary:
            test = summary[test_key]
            sig = "YES" if test["significant"] else "NO"
            print(f"\nMcNemar's test ({label}):")
            print(f"  Statistic: {test['statistic']:.4f}")
            print(f"  p-value:   {test['p_value']:.4f}")
            print(f"  Significant (alpha=0.05): {sig}")

    print("=" * 60)


def generate_results_figure(summary: dict, output_path: str = None):
    """
    Generate a bar chart comparing method accuracies.
    Saves to output_path or displays.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Skipping figure generation.")
        return

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
                                   "results_comparison.png")
    plt.savefig(output_path, dpi=150)
    print(f"Figure saved to: {output_path}")
    plt.close()


def generate_heatmap(debate_results: list[dict] = None,
                     direct_qa_results: list[dict] = None,
                     self_consistency_results: list[dict] = None,
                     output_path: str = None):
    """
    Generate a per-question heatmap showing correct/incorrect across all three methods.
    Questions on one axis, methods on the other.
    Saves to output_path or a default file.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not available. Skipping heatmap generation.")
        return

    # Load from logs if not provided
    if debate_results is None:
        debate_log = load_latest_log("debate_")
        debate_results = debate_log.get("results", []) if debate_log else []

    if direct_qa_results is None:
        direct_qa_log = load_latest_log("baseline_direct_qa_")
        direct_qa_results = direct_qa_log.get("results", []) if direct_qa_log else []

    if self_consistency_results is None:
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
                                   "results_heatmap.png")
    plt.savefig(output_path, dpi=150)
    print(f"Heatmap saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    summary = evaluate_all()
    print_evaluation_table(summary)
    generate_results_figure(summary)
    generate_heatmap()
