# LLM Debate with Judge Pipeline

An experiment in whether structured debate between LLMs improves factual accuracy on yes/no reasoning questions. Two Claude debaters argue over [StrategyQA](https://huggingface.co/datasets/wics/strategy-qa) questions, a judge evaluates the debate, and the final answer is compared to ground truth. Results are benchmarked against two baselines: direct chain-of-thought QA and self-consistency (majority vote over multiple samples).

Built for UTSA CS 7347 (NLP), Spring 2026, Assignment 2.

## Setup

1. **Python 3.10+** (tested on 3.14)

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root:
   ```
   ANTHROPIC_API_KEY=your-key-here
   ```

## Usage

### Web UI

```bash
python ui.py
```

Opens a Gradio interface with:
- **Single Question mode** — fetch a random StrategyQA question or type your own, run one debate
- **Full StrategyQA Sample mode** — batch process `sample_size` questions with a progress bar
- Controls for model selection, temperature, max tokens, number of rounds (3–5), and judge count (1 or 3)
- Animated SVG status display (debating podiums, gavel verdict animation)

### CLI

```bash
# Run the full debate pipeline on the configured sample
python orchestrator.py

# Run baselines (direct QA + self-consistency)
python baselines.py
```

### Evaluation

```bash
python evaluation.py
```

Loads the most recent debate and baseline logs, computes accuracy for each method, and runs McNemar's test for statistical comparison.

## How It Works

### Pipeline (4 Phases)

1. **Initialization** — Both debaters independently generate an initial position on the question. If they agree immediately, the debate is skipped.

2. **Multi-round Debate** — Debater A argues first, then Debater B responds with A's argument as context. Each debater maintains a multi-turn conversation history (system prompt + alternating user/assistant messages) so the LLM sees the full debate context each round. Stops early if both debaters agree for 2 consecutive rounds.

3. **Judgment** — A judge LLM reads the full debate transcript and produces: chain-of-thought analysis, strongest/weakest arguments per side, a verdict (Debater A or B), the final answer (Yes/No), and a confidence score (1–5). Optionally uses 3 judges with majority vote.

4. **Evaluation** — The judge's answer is compared to ground truth. All intermediate data is logged as JSON.

### Baselines

- **Direct QA** — Single chain-of-thought prompt, no debate. Temperature 0 for deterministic output.
- **Self-Consistency** — 5 samples at temperature 0.7, majority vote on Yes/No.

## Supported Models & Value Ranges

**Models:** `claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-6`

| Parameter | Range | Notes |
|-----------|-------|-------|
| `temperature` | 0.0–1.0 | 0.0 = deterministic, 0.7 = default for debaters |
| `max_tokens` | 300–800 | Debater response length |
| `judge_max_tokens` | 300–1500 | Judge response length |
| `num_rounds` | 3–5 | Debate rounds before forced stop |
| `num_judges` | 1 or 3 | 3 = majority vote |
| `sample_size` | 1–500 | Questions per batch run (max = cached dataset size) |
| `sample_count` | 1, 3, 5, 7... | Self-consistency samples (odd only) |

## Configuration

All parameters are in `config/config.json`:

```json
{
    "debate": {
        "debater_a_model": "claude-haiku-4-5",
        "debater_b_model": "claude-haiku-4-5",
        "judge_model": "claude-sonnet-4-6",
        "debater_a_temperature": 0.7,
        "debater_b_temperature": 0.7,
        "judge_temperature": 0.3,
        "max_tokens": 500,
        "judge_max_tokens": 800,
        "num_rounds": 3,
        "num_judges": 1,
        "sample_size": 100
    },
    "direct_qa": { ... },
    "self_consistency": { ... }
}
```

## Project Structure

```
orchestrator.py     Pipeline orchestration (4-phase flow)
debater.py          Multi-turn debate message building
judge.py            Debate evaluation + structured response parsing
api_utils.py        Anthropic API wrapper with retry logic
data.py             StrategyQA fetching + caching
evaluation.py       Accuracy computation + McNemar's test
baselines.py        Direct QA and self-consistency baselines
logging_utils.py    JSON log writing
ui.py               Gradio web interface
config/             Experiment parameters (config.json)
data/               Cached dataset (strategyqa_cache.json)
prompts/            System prompt templates (.txt)
logs/               JSON transcripts (auto-created)
```

## Data

Questions come from [StrategyQA](https://huggingface.co/datasets/wics/strategy-qa) via the HuggingFace HTTP API (no `datasets` library needed). On first run, 500 questions are fetched and saved to `data/strategyqa_cache.json`. Subsequent runs load from this file, ensuring the same questions are used across all sessions.
