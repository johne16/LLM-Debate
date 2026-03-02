"""
Evaluation module: Compare debate pipeline vs baselines.
Takes 6 log files (4 debate + 2 baselines) for a single temperature group.
Generates a combined bar chart, heatmap, CSV summary, and McNemar's tests.
"""

import csv
import glob
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy import stats


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def load_log(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_debate_label(config: dict) -> str:
    """Build a label from debate log config (e.g. 'Sonnet 3-Judge')."""
    judge_model = config.get("judge_model", "")
    num_judges = config.get("num_judges", 1)

    if "sonnet" in judge_model:
        model_name = "Sonnet"
    elif "haiku" in judge_model:
        model_name = "Haiku"
    elif "opus" in judge_model:
        model_name = "Opus"
    else:
        model_name = judge_model

    judge_str = f"{num_judges}-Judge" if num_judges > 1 else "1-Judge"
    return f"{model_name} {judge_str}"


def compute_accuracy(results: list[dict]) -> float:
    scored = [r for r in results if r.get("correct") is not None]
    if not scored:
        return 0.0
    return sum(1 for r in scored if r["correct"]) / len(scored)


def mcnemar_test(results_a: list[bool], results_b: list[bool]) -> dict:
    assert len(results_a) == len(results_b), "Result lists must be same length"

    a_right_b_wrong = sum(1 for a, b in zip(results_a, results_b) if a and not b)
    a_wrong_b_right = sum(1 for a, b in zip(results_a, results_b) if not a and b)

    if a_right_b_wrong + a_wrong_b_right == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False,
                "a_right_b_wrong": 0, "a_wrong_b_right": 0}

    statistic = (abs(a_right_b_wrong - a_wrong_b_right) - 1) ** 2 / (a_right_b_wrong + a_wrong_b_right)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "a_right_b_wrong": a_right_b_wrong,
        "a_wrong_b_right": a_wrong_b_right
    }


def evaluate(debate_logs: list[dict], baseline_logs: list[dict]) -> dict:
    """
    Build evaluation summary from loaded logs.

    Args:
        debate_logs: List of 4 debate log dicts.
        baseline_logs: List of 2 baseline log dicts (direct_qa, self_consistency).

    Returns:
        Dict with per-method accuracy, labels, McNemar results, and configs.
    """
    methods = []

    for log in debate_logs:
        config = log.get("config", {})
        results = log.get("results", [])
        scored = [r for r in results if r.get("correct") is not None]
        label = get_debate_label(config)

        correct_confidences = []
        incorrect_confidences = []
        consensus_count = 0
        for r in scored:
            if r.get("consensus_skip"):
                consensus_count += 1
            judge_result = r.get("judge_result", {})
            confidence = judge_result.get("avg_confidence")
            if confidence is not None:
                if r["correct"]:
                    correct_confidences.append(confidence)
                else:
                    incorrect_confidences.append(confidence)

        methods.append({
            "label": label,
            "type": "debate",
            "accuracy": compute_accuracy(results),
            "total": len(scored),
            "correct": sum(1 for r in scored if r["correct"]),
            "config": config,
            "scored": scored,
            "consensus_rate": consensus_count / len(scored) if scored else 0.0,
            "consensus_count": consensus_count,
            "avg_confidence_correct": (
                sum(correct_confidences) / len(correct_confidences) if correct_confidences else None
            ),
            "avg_confidence_incorrect": (
                sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else None
            ),
        })

    for log in baseline_logs:
        config = log.get("config", {})
        results = log.get("results", [])
        scored = [r for r in results if r.get("correct") is not None]
        method_name = log.get("method", "unknown")
        if method_name == "direct_qa":
            label = "Direct QA"
        elif method_name == "self_consistency":
            label = "Self-Consistency"
        else:
            label = method_name

        methods.append({
            "label": label,
            "type": "baseline",
            "accuracy": compute_accuracy(results),
            "total": len(scored),
            "correct": sum(1 for r in scored if r["correct"]),
            "config": config,
            "scored": scored,
        })

    # McNemar tests: each debate config vs each baseline
    mcnemar_results = []
    for m in methods:
        if m["type"] != "debate":
            continue
        for b in methods:
            if b["type"] != "baseline":
                continue
            if len(m["scored"]) == len(b["scored"]) and len(m["scored"]) > 0:
                a_correct = [bool(r["correct"]) for r in m["scored"]]
                b_correct = [bool(r["correct"]) for r in b["scored"]]
                test = mcnemar_test(a_correct, b_correct)
                mcnemar_results.append({
                    "comparison": f"{m['label']} vs {b['label']}",
                    **test
                })

    return {"methods": methods, "mcnemar": mcnemar_results}


def save_csv(summary: dict, output_path: str = None):
    if output_path is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        output_path = os.path.join(RESULTS_DIR, f"results_summary_{TIMESTAMP}.csv")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Correct", "Total", "Accuracy"])
        for m in summary["methods"]:
            writer.writerow([m["label"], m["correct"], m["total"], f"{m['accuracy']:.4f}"])

        # Debate-specific metrics
        writer.writerow([])
        writer.writerow(["Debate Metrics"])
        writer.writerow(["Method", "Consensus Rate", "Consensus Count",
                          "Avg Confidence (Correct)", "Avg Confidence (Incorrect)"])
        for m in summary["methods"]:
            if m["type"] != "debate":
                continue
            writer.writerow([
                m["label"],
                f"{m.get('consensus_rate', 0):.4f}",
                m.get("consensus_count", 0),
                f"{m['avg_confidence_correct']:.2f}" if m.get("avg_confidence_correct") is not None else "N/A",
                f"{m['avg_confidence_incorrect']:.2f}" if m.get("avg_confidence_incorrect") is not None else "N/A",
            ])

        # McNemar results
        if summary["mcnemar"]:
            writer.writerow([])
            writer.writerow(["McNemar's Test Results"])
            writer.writerow(["Comparison", "Statistic", "p-value", "Significant (alpha=0.05)",
                             "A Right/B Wrong", "A Wrong/B Right"])
            for t in summary["mcnemar"]:
                writer.writerow([
                    t["comparison"],
                    f"{t['statistic']:.4f}",
                    f"{t['p_value']:.4f}",
                    "YES" if t["significant"] else "NO",
                    t["a_right_b_wrong"],
                    t["a_wrong_b_right"]
                ])

        # Configuration
        writer.writerow([])
        writer.writerow(["Configuration"])
        writer.writerow(["Method", "Parameter", "Value"])
        for m in summary["methods"]:
            for param, value in m["config"].items():
                writer.writerow([m["label"], param, value])

    print(f"CSV saved to: {output_path}")


def generate_bar_chart(summary: dict, output_path: str = None):
    methods = summary["methods"]
    labels = [m["label"] for m in methods]
    accuracies = [m["accuracy"] * 100 for m in methods]

    colors = []
    for m in methods:
        if m["type"] == "baseline":
            colors.append("#FF9800" if m["label"] == "Direct QA" else "#4CAF50")
        else:
            colors.append("#2196F3")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), accuracies, color=colors, edgecolor="black", linewidth=0.8)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title("Accuracy Comparison")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        output_path = os.path.join(RESULTS_DIR, f"results_comparison_{TIMESTAMP}.png")
    plt.savefig(output_path, dpi=150)
    print(f"Bar chart saved to: {output_path}")
    plt.close()


def generate_heatmap(summary: dict, output_path: str = None):
    methods = summary["methods"]

    num_questions = min(len(m["scored"]) for m in methods)
    if num_questions == 0:
        print("Not enough matched results across methods for heatmap.")
        return

    num_methods = len(methods)
    data = np.zeros((num_questions, num_methods))
    for col, m in enumerate(methods):
        for row in range(num_questions):
            data[row, col] = 1 if m["scored"][row]["correct"] else 0

    fig, ax = plt.subplots(figsize=(max(5, num_methods * 1.5), max(4, num_questions * 0.35)))

    cmap = mcolors.ListedColormap(["#EF5350", "#66BB6A"])
    ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(np.arange(-0.5, num_methods, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_questions, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    q_labels = [f"Q{i+1}" for i in range(num_questions)]
    method_labels = [m["label"] for m in methods]

    ax.set_xticks(range(num_methods))
    ax.set_xticklabels(method_labels, fontsize=8, rotation=15, ha="right")
    ax.set_yticks(range(num_questions))
    ax.set_yticklabels(q_labels, fontsize=8)
    ax.set_title("Per-Question Correctness")

    plt.tight_layout()

    if output_path is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        output_path = os.path.join(RESULTS_DIR, f"results_heatmap_{TIMESTAMP}.png")
    plt.savefig(output_path, dpi=150)
    print(f"Heatmap saved to: {output_path}")
    plt.close()


LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(LOGS_DIR, "*.json")))
    if not files:
        print(f"No JSON files found in {LOGS_DIR}")
        exit(1)

    debate_files = [f for f in files if "baseline" not in os.path.basename(f)]
    baseline_files = [f for f in files if "baseline" in os.path.basename(f)]

    if len(debate_files) != 4:
        print(f"Expected 4 debate logs, found {len(debate_files)}")
        exit(1)
    if len(baseline_files) != 2:
        print(f"Expected 2 baseline logs, found {len(baseline_files)}")
        exit(1)

    debate_logs = [load_log(p) for p in debate_files]
    baseline_logs = [load_log(p) for p in baseline_files]

    summary = evaluate(debate_logs, baseline_logs)
    save_csv(summary)
    generate_bar_chart(summary)
    generate_heatmap(summary)
