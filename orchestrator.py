"""
Orchestrator: Runs the 4-phase Debate + Judge pipeline.

Phase 1: Initialization (independent initial positions, consensus check)
Phase 2: Multi-round debate (adaptive stopping)
Phase 3: Judgment (single or multi-judge)
Phase 4: Evaluation (compare to ground truth, log everything)
"""

import json
import os
import anthropic
from dotenv import load_dotenv

from data import fetch_strategy_qa
from debater import generate_initial_position, generate_debate_argument
from judge import evaluate_debate, evaluate_debate_multi
from logging_utils import save_debate_log, save_debate_batch_log


def load_config(config_path: str = None) -> dict:
    """Load configuration from JSON file."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_transcript(initial_positions: dict, rounds: list) -> str:
    """
    Build a formatted debate transcript string from initial positions and rounds.
    Used only for the judge's evaluation.
    """
    lines = []

    lines.append("=== INITIAL POSITIONS ===")
    lines.append(f"\nDebater A (Initial Position):")
    lines.append(f"Answer: {initial_positions['debater_a']['answer']}")
    lines.append(initial_positions['debater_a']['reasoning'])

    lines.append(f"\nDebater B (Initial Position):")
    lines.append(f"Answer: {initial_positions['debater_b']['answer']}")
    lines.append(initial_positions['debater_b']['reasoning'])

    if rounds:
        lines.append("\n=== DEBATE ROUNDS ===")
        for i, round_data in enumerate(rounds, 1):
            lines.append(f"\n--- Round {i} ---")
            lines.append(f"\nDebater A (Round {i}):")
            lines.append(f"Answer: {round_data['debater_a']['answer']}")
            lines.append(round_data['debater_a']['argument'])
            lines.append(f"\nDebater B (Round {i}):")
            lines.append(f"Answer: {round_data['debater_b']['answer']}")
            lines.append(round_data['debater_b']['argument'])

    return "\n".join(lines)


def run_single_debate(client: anthropic.Anthropic, question: str,
                      ground_truth, config: dict,
                      cancel_event=None, batch=False) -> dict:
    """
    Run the full 4-phase debate pipeline for a single question.

    Args:
        client: Anthropic client.
        question: The question to debate.
        ground_truth: True/False for StrategyQA questions, or None for custom.
        config: Config dict.
        cancel_event: Optional threading.Event to signal cancellation.

    Returns:
        Dict with all debate data and evaluation result.
    """
    debate_cfg = config["debate"]

    # ---- Phase 1: Initialization ----
    print(f"\n  Phase 1: Generating initial positions...")

    if cancel_event and cancel_event.is_set():
        return {"cancelled": True, "question": question}

    position_a = generate_initial_position(
        client, question, "A",
        debate_cfg["debater_a_model"],
        debate_cfg["debater_a_temperature"],
        debate_cfg["max_tokens"]
    )

    if cancel_event and cancel_event.is_set():
        return {"cancelled": True, "question": question}

    position_b = generate_initial_position(
        client, question, "B",
        debate_cfg["debater_b_model"],
        debate_cfg["debater_b_temperature"],
        debate_cfg["max_tokens"]
    )

    if cancel_event and cancel_event.is_set():
        return {"cancelled": True, "question": question}

    initial_positions = {"debater_a": position_a, "debater_b": position_b}
    print(f"    Debater A: {position_a['answer']}  |  Debater B: {position_b['answer']}")

    rounds = []
    consensus_skip = False

    # Check for initial consensus
    if position_a["answer"] == position_b["answer"] and position_a["answer"] != "Unknown":
        print(f"  Consensus reached at initialization: {position_a['answer']}. Skipping debate.")
        consensus_skip = True
    else:
        # ---- Phase 2: Multi-round Debate ----
        num_rounds = debate_cfg["num_rounds"]
        current_answer_a = position_a["answer"]
        current_answer_b = position_b["answer"]
        consecutive_agreement = 0

        for round_num in range(1, num_rounds + 1):
            print(f"  Phase 2: Round {round_num}/{num_rounds}...")

            if cancel_event and cancel_event.is_set():
                return {"cancelled": True, "question": question}

            # Debater A argues (no opponent argument yet for this round)
            argument_a = generate_debate_argument(
                client, question, "A", current_answer_a,
                initial_positions, rounds,
                opponent_current_argument="",
                model=debate_cfg["debater_a_model"],
                temperature=debate_cfg["debater_a_temperature"],
                max_tokens=debate_cfg["max_tokens"]
            )

            if cancel_event and cancel_event.is_set():
                return {"cancelled": True, "question": question}

            # Debater B responds (with A's current-round argument)
            argument_b = generate_debate_argument(
                client, question, "B", current_answer_b,
                initial_positions, rounds,
                opponent_current_argument=argument_a["raw_response"],
                model=debate_cfg["debater_b_model"],
                temperature=debate_cfg["debater_b_temperature"],
                max_tokens=debate_cfg["max_tokens"]
            )

            if cancel_event and cancel_event.is_set():
                return {"cancelled": True, "question": question}

            round_data = {"debater_a": argument_a, "debater_b": argument_b}
            rounds.append(round_data)

            current_answer_a = argument_a["answer"]
            current_answer_b = argument_b["answer"]

            print(f"    Debater A: {current_answer_a}  |  Debater B: {current_answer_b}")

            # Adaptive stopping: check for consecutive agreement
            if current_answer_a == current_answer_b and current_answer_a != "Unknown":
                consecutive_agreement += 1
                if consecutive_agreement >= 2:
                    print(f"  Adaptive stop: Both debaters agreed for 2 consecutive rounds.")
                    break
            else:
                consecutive_agreement = 0

    # ---- Phase 3: Judgment ----
    print(f"  Phase 3: Judge evaluating debate...")

    if cancel_event and cancel_event.is_set():
        return {"cancelled": True, "question": question}

    # Build text transcript only for the judge
    transcript = build_transcript(initial_positions, rounds)

    num_judges = debate_cfg.get("num_judges", 1)
    debater_a_final = rounds[-1]["debater_a"]["answer"] if rounds else position_a["answer"]
    debater_b_final = rounds[-1]["debater_b"]["answer"] if rounds else position_b["answer"]

    if num_judges == 1:
        judge_result = evaluate_debate(
            client, question, transcript,
            debate_cfg["judge_model"],
            debate_cfg["judge_temperature"],
            debate_cfg["judge_max_tokens"],
            debater_a_answer=debater_a_final,
            debater_b_answer=debater_b_final
        )
        # Copy BEFORE adding aggregation keys (fix #12)
        individual_copy = judge_result.copy()
        judge_result["final_answer"] = judge_result["answer"]
        # Normalize verdict to match multi-judge format
        if debater_a_final == debater_b_final:
            judge_result["final_verdict"] = "Agreed"
        elif judge_result["answer"] == debater_a_final:
            judge_result["final_verdict"] = "Debater A"
        elif judge_result["answer"] == debater_b_final:
            judge_result["final_verdict"] = "Debater B"
        else:
            judge_result["final_verdict"] = judge_result.get("verdict", "Unknown")
        judge_result["avg_confidence"] = judge_result["confidence"]
        judge_result["individual_judgments"] = [individual_copy]
    else:
        judge_result = evaluate_debate_multi(
            client, question, transcript,
            debater_a_final, debater_b_final,
            debate_cfg["judge_model"],
            debate_cfg["judge_temperature"],
            debate_cfg["judge_max_tokens"],
            num_judges
        )

    # ---- Phase 4: Evaluation ----
    final_answer = judge_result.get("final_answer", "Unknown")

    if ground_truth is None:
        ground_truth_label = "N/A"
        correct = None
    else:
        ground_truth_label = "Yes" if ground_truth else "No"
        correct = final_answer == ground_truth_label

    print(f"  Phase 4: Judge answer = {final_answer} | "
          f"Ground truth = {ground_truth_label} | "
          f"Correct = {correct}")

    if not batch:
        save_debate_log(
            question=question,
            ground_truth=ground_truth,
            initial_positions=initial_positions,
            rounds=rounds,
            judge_result=judge_result,
            config=config,
            consensus_skip=consensus_skip
        )

    return {
        "question": question,
        "ground_truth": ground_truth,
        "ground_truth_label": ground_truth_label,
        "initial_positions": initial_positions,
        "rounds": rounds,
        "judge_result": judge_result,
        "final_answer": final_answer,
        "correct": correct,
        "consensus_skip": consensus_skip,
    }


def run_debate_pipeline(config_path: str = None) -> dict:
    """
    Run the full debate pipeline on all sampled StrategyQA questions.

    Returns:
        Dict with 'results' (list) and 'accuracy' (float).
    """
    load_dotenv()
    config = load_config(config_path)
    client = anthropic.Anthropic()

    sample_size = config["debate"]["sample_size"]
    print(f"Fetching {sample_size} StrategyQA questions...")
    questions = fetch_strategy_qa(sample_size)
    print(f"Fetched {len(questions)} questions.\n")

    results = []
    for i, q in enumerate(questions):
        print(f"=== Question {i + 1}/{len(questions)} ===")
        print(f"  Q: {q['question']}")
        result = run_single_debate(client, q["question"], q["answer"], config, batch=True)
        results.append(result)
        print()

    scored = [r for r in results if r.get("correct") is not None]
    correct_count = sum(1 for r in scored if r["correct"])
    accuracy = correct_count / len(scored) if scored else 0.0
    log_path = save_debate_batch_log(results, config)

    print(f"\n{'='*50}")
    print(f"DEBATE PIPELINE RESULTS")
    print(f"  Total questions: {len(results)}")
    print(f"  Correct: {correct_count}")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Log saved: {log_path}")
    print(f"{'='*50}")

    return {"results": results, "accuracy": accuracy}


if __name__ == "__main__":
    run_debate_pipeline()
