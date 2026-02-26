"""
Web UI for the LLM Debate Pipeline using Gradio.

Provides controls for model selection, temperature, max tokens,
number of rounds, and number of judges. Supports two run modes:
single question and full StrategyQA sample.

Full debate mode includes animated SVG status display:
- Debating state: two podiums with microphones and wiggling chat bubbles
- Verdict state: banging gavel with radiating impact lines
- Result text: winner announcement before transitioning to next question
"""

import os
import time
import threading
import anthropic
import gradio as gr
from dotenv import load_dotenv

from data import fetch_strategy_qa, fetch_random_question as fetch_random_from_cache
from orchestrator import run_single_debate, load_config


load_dotenv()

MODEL_CHOICES = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-6"]

# Threading event used to signal a running debate to stop.
_cancel_event = threading.Event()


# ---------------------------------------------------------------------------
# SVG animation HTML builders
# ---------------------------------------------------------------------------

def _svg_debating(question_text: str, question_num: int, total: int) -> str:
    """Return inline HTML/SVG for the 'Debating...' animation state."""
    escaped_q = (question_text
                 .replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;")
                 .replace('"', "&quot;"))
    return f'''
<div id="debate-anim-wrapper" style="text-align:center; padding:20px 0;">
  <style>
    @keyframes wiggle {{
      0%   {{ transform: translateX(0) rotate(0deg); }}
      25%  {{ transform: translateX(-3px) rotate(-3deg); }}
      50%  {{ transform: translateX(3px) rotate(3deg); }}
      75%  {{ transform: translateX(-2px) rotate(-2deg); }}
      100% {{ transform: translateX(0) rotate(0deg); }}
    }}
    @keyframes float-up {{
      0%   {{ transform: translateY(0); opacity: 1; }}
      100% {{ transform: translateY(-8px); opacity: 0.7; }}
    }}
    .bubble-a {{ animation: wiggle 0.8s ease-in-out infinite; transform-origin: center bottom; }}
    .bubble-b {{ animation: wiggle 0.8s ease-in-out infinite 0.4s; transform-origin: center bottom; }}
  </style>

  <div style="font-size:14px; color:#666; margin-bottom:6px;">
    Question {question_num} / {total}
  </div>
  <div style="font-size:15px; font-weight:600; color:#333; margin-bottom:14px;
              max-width:500px; margin-left:auto; margin-right:auto;">
    {escaped_q}
  </div>

  <svg width="420" height="220" viewBox="0 0 420 220" xmlns="http://www.w3.org/2000/svg"
       style="display:block; margin:0 auto;">

    <!-- Left podium (trapezoid) -->
    <polygon points="40,200 120,200 110,140 50,140"
             fill="none" stroke="#444" stroke-width="2.5" stroke-linejoin="round"/>
    <!-- Left microphone: stand + circle -->
    <line x1="80" y1="140" x2="80" y2="110" stroke="#444" stroke-width="2" stroke-linecap="round"/>
    <circle cx="80" cy="105" r="7" fill="none" stroke="#444" stroke-width="2"/>

    <!-- Left chat bubble (wiggling) -->
    <g class="bubble-a">
      <rect x="50" y="50" width="60" height="36" rx="10" ry="10"
            fill="none" stroke="#2563eb" stroke-width="2"/>
      <polygon points="72,86 80,96 88,86" fill="none" stroke="#2563eb" stroke-width="2"
               stroke-linejoin="round"/>
      <!-- Dots inside bubble -->
      <circle cx="68" cy="68" r="3" fill="#2563eb"/>
      <circle cx="80" cy="68" r="3" fill="#2563eb"/>
      <circle cx="92" cy="68" r="3" fill="#2563eb"/>
    </g>

    <!-- Label A -->
    <text x="80" y="175" text-anchor="middle" font-size="16" font-weight="bold"
          fill="#2563eb" font-family="sans-serif">A</text>

    <!-- Right podium (trapezoid) -->
    <polygon points="300,200 380,200 370,140 310,140"
             fill="none" stroke="#444" stroke-width="2.5" stroke-linejoin="round"/>
    <!-- Right microphone: stand + circle -->
    <line x1="340" y1="140" x2="340" y2="110" stroke="#444" stroke-width="2" stroke-linecap="round"/>
    <circle cx="340" cy="105" r="7" fill="none" stroke="#444" stroke-width="2"/>

    <!-- Right chat bubble (wiggling, offset) -->
    <g class="bubble-b">
      <rect x="310" y="50" width="60" height="36" rx="10" ry="10"
            fill="none" stroke="#dc2626" stroke-width="2"/>
      <polygon points="332,86 340,96 348,86" fill="none" stroke="#dc2626" stroke-width="2"
               stroke-linejoin="round"/>
      <!-- Dots inside bubble -->
      <circle cx="328" cy="68" r="3" fill="#dc2626"/>
      <circle cx="340" cy="68" r="3" fill="#dc2626"/>
      <circle cx="352" cy="68" r="3" fill="#dc2626"/>
    </g>

    <!-- Label B -->
    <text x="340" y="175" text-anchor="middle" font-size="16" font-weight="bold"
          fill="#dc2626" font-family="sans-serif">B</text>

    <!-- VS text -->
    <text x="210" y="160" text-anchor="middle" font-size="20" font-weight="bold"
          fill="#888" font-family="sans-serif">VS</text>
  </svg>

  <div style="font-size:16px; font-weight:600; color:#555; margin-top:10px;">
    Debating...
  </div>
</div>
'''


def _svg_gavel(result_text: str) -> str:
    """Return inline HTML/SVG for the verdict gavel animation."""
    escaped = (result_text
               .replace("&", "&amp;")
               .replace("<", "&lt;")
               .replace(">", "&gt;"))
    return f'''
<div id="gavel-anim-wrapper" style="text-align:center; padding:20px 0;">
  <style>
    @keyframes gavel-bang {{
      0%   {{ transform: rotate(-30deg); }}
      40%  {{ transform: rotate(8deg); }}
      50%  {{ transform: rotate(8deg); }}
      70%  {{ transform: rotate(-30deg); }}
      100% {{ transform: rotate(-30deg); }}
    }}
    @keyframes impact-flash {{
      0%   {{ opacity: 0; }}
      40%  {{ opacity: 0; }}
      50%  {{ opacity: 1; }}
      70%  {{ opacity: 0.3; }}
      100% {{ opacity: 0; }}
    }}
    @keyframes result-pop {{
      0%   {{ transform: scale(0.5); opacity: 0; }}
      60%  {{ transform: scale(1.1); opacity: 1; }}
      100% {{ transform: scale(1.0); opacity: 1; }}
    }}
    .gavel-head {{
      animation: gavel-bang 1.0s ease-in-out 3;
      transform-origin: 210px 150px;
    }}
    .impact-lines {{
      animation: impact-flash 1.0s ease-in-out 3;
    }}
    .result-label {{
      opacity: 1;
    }}
  </style>

  <svg width="420" height="200" viewBox="0 0 420 200" xmlns="http://www.w3.org/2000/svg"
       style="display:block; margin:0 auto;">

    <!-- Sound block / base -->
    <rect x="160" y="155" width="100" height="20" rx="3" ry="3"
          fill="none" stroke="#444" stroke-width="2.5"/>

    <!-- Gavel group (handle + head), rotated 90 clockwise, pivots at base -->
    <g class="gavel-head">
      <g transform="rotate(90, 210, 110)">
        <!-- Handle -->
        <rect x="205" y="70" width="10" height="80" rx="2" ry="2"
              fill="none" stroke="#444" stroke-width="2.5"/>
        <!-- Head (wide rectangle across handle top) -->
        <rect x="180" y="55" width="60" height="22" rx="4" ry="4"
              fill="none" stroke="#444" stroke-width="2.5"/>
      </g>
    </g>

    <!-- Impact / radiating lines (visible on downstroke) -->
    <g class="impact-lines">
      <line x1="150" y1="152" x2="135" y2="140" stroke="#b91c1c" stroke-width="2" stroke-linecap="round"/>
      <line x1="145" y1="165" x2="125" y2="165" stroke="#b91c1c" stroke-width="2" stroke-linecap="round"/>
      <line x1="150" y1="178" x2="135" y2="190" stroke="#b91c1c" stroke-width="2" stroke-linecap="round"/>
      <line x1="270" y1="152" x2="285" y2="140" stroke="#b91c1c" stroke-width="2" stroke-linecap="round"/>
      <line x1="275" y1="165" x2="295" y2="165" stroke="#b91c1c" stroke-width="2" stroke-linecap="round"/>
      <line x1="270" y1="178" x2="285" y2="190" stroke="#b91c1c" stroke-width="2" stroke-linecap="round"/>
    </g>
  </svg>

  <div class="result-label"
       style="font-size:22px; font-weight:700; color:currentColor; margin-top:6px;
              font-family:sans-serif;">
    {escaped}
  </div>
</div>
'''


def _svg_result_static(result_text: str) -> str:
    """Return static HTML showing the final verdict (no animation)."""
    escaped = (result_text
               .replace("&", "&amp;")
               .replace("<", "&lt;")
               .replace(">", "&gt;"))
    return f'''
<div style="text-align:center; padding:40px 0;">
  <div style="font-size:28px; font-weight:700; color:currentColor; font-family:sans-serif;">
    {escaped}
  </div>
</div>
'''


def _svg_empty() -> str:
    """Return empty/cleared HTML for the animation area."""
    return '<div style="min-height:40px;"></div>'


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _build_config_from_ui(debater_a_model, debater_b_model, judge_model,
                           temp_a, temp_b, temp_judge, max_tokens,
                           judge_max_tokens, num_rounds, num_judges):
    """Build a config dict matching config.json structure from UI values."""
    base = load_config()
    base["debate"].update({
        "debater_a_model": debater_a_model,
        "debater_b_model": debater_b_model,
        "judge_model": judge_model,
        "debater_a_temperature": float(temp_a),
        "debater_b_temperature": float(temp_b),
        "judge_temperature": float(temp_judge),
        "max_tokens": int(max_tokens),
        "judge_max_tokens": int(judge_max_tokens),
        "num_rounds": int(num_rounds),
        "num_judges": int(num_judges),
    })
    return base


# ---------------------------------------------------------------------------
# Run functions
# ---------------------------------------------------------------------------

def run_single_question_ui(question: str, ground_truth,
                           debater_a_model: str, debater_b_model: str,
                           judge_model: str, temp_a: float, temp_b: float,
                           temp_judge: float, max_tokens: int, judge_max_tokens: int,
                           num_rounds: int, num_judges: int):
    """Run a single debate from the UI with animations."""
    _cancel_event.clear()
    config = _build_config_from_ui(
        debater_a_model, debater_b_model, judge_model,
        temp_a, temp_b, temp_judge, max_tokens, judge_max_tokens,
        num_rounds, num_judges
    )
    client = anthropic.Anthropic()

    # Show debating animation
    yield 0, _svg_debating(question, 1, 1)

    result = run_single_debate(client, question, ground_truth, config,
                               cancel_event=_cancel_event)

    if result.get("cancelled"):
        yield 0, _svg_empty()
        return

    if _cancel_event.is_set():
        return

    # Show gavel + verdict, then hold static result
    result_text = _determine_result_text(result)
    yield 1.0, _svg_gavel(result_text)
    time.sleep(4)
    yield 1.0, _svg_result_static(result_text)


def _determine_result_text(result: dict) -> str:
    """Determine the display text for a debate result."""
    if result.get("consensus_skip"):
        return "Agreed!"
    verdict = result.get("judge_result", {}).get("final_verdict", "")
    if "Debater A" in verdict:
        return "Winner: Debater A!"
    elif "Debater B" in verdict:
        return "Winner: Debater B!"
    # Fallback: check if both debaters ended up agreeing
    return "Agreed!"


def run_full_debate_ui(debater_a_model: str, debater_b_model: str,
                       judge_model: str, temp_a: float, temp_b: float,
                       temp_judge: float, max_tokens: int, judge_max_tokens: int,
                       num_rounds: int, num_judges: int):
    """
    Run the full StrategyQA debate pipeline from the UI.

    This is a generator that yields (progress_fraction, animation_html) tuples
    to update the progress bar and SVG animation area in real time.
    """
    _cancel_event.clear()
    config = _build_config_from_ui(
        debater_a_model, debater_b_model, judge_model,
        temp_a, temp_b, temp_judge, max_tokens, judge_max_tokens,
        num_rounds, num_judges
    )
    sample_size = config["debate"]["sample_size"]
    client = anthropic.Anthropic()

    questions = fetch_strategy_qa(sample_size)
    total = len(questions)

    for i, q in enumerate(questions):
        if _cancel_event.is_set():
            return

        # --- Show debating animation ---
        progress = i / total
        yield progress, _svg_debating(q["question"], i + 1, total)

        # --- Run the actual debate ---
        result = run_single_debate(client, q["question"], q["answer"], config,
                                   cancel_event=_cancel_event)

        if result.get("cancelled") or _cancel_event.is_set():
            return

        # --- Show gavel / verdict animation ---
        result_text = _determine_result_text(result)
        verdict_progress = (i + 0.8) / total
        yield verdict_progress, _svg_gavel(result_text)
        time.sleep(4)
        yield verdict_progress, _svg_result_static(result_text)
        time.sleep(2)

        if _cancel_event.is_set():
            return

    # All done
    yield 1.0, _svg_empty()


def run_debate_dispatch(mode: str, question: str, ground_truth_input: str,
                        debater_a_model: str,
                        debater_b_model: str, judge_model: str,
                        temp_a: float, temp_b: float, temp_judge: float,
                        max_tokens: int, judge_max_tokens: int,
                        num_rounds: int, num_judges: int):
    """
    Dispatch to the correct run function based on mode selection.

    For full debate mode this is a generator yielding (progress, html) tuples.
    For single question mode it runs synchronously and yields nothing meaningful.
    """
    if mode == "Full StrategyQA Sample":
        yield from run_full_debate_ui(
            debater_a_model, debater_b_model, judge_model,
            temp_a, temp_b, temp_judge, max_tokens, judge_max_tokens,
            num_rounds, num_judges
        )
    else:
        if not question or not question.strip():
            return
        if not ground_truth_input:
            return
        ground_truth = ground_truth_input == "Yes"
        yield from run_single_question_ui(
            question, ground_truth, debater_a_model, debater_b_model, judge_model,
            temp_a, temp_b, temp_judge, max_tokens, judge_max_tokens,
            num_rounds, num_judges
        )


def fetch_random_question():
    """Fetch a random StrategyQA question for the UI."""
    question = fetch_random_from_cache()
    ground_truth = "Yes" if question["answer"] else "No"
    return question["question"], ground_truth


def stop_and_quit():
    """Signal any running debate to stop and schedule Gradio server shutdown."""
    _cancel_event.set()
    # Schedule server shutdown on a short timer so this response can complete first.
    def _shutdown():
        time.sleep(1)
        os._exit(0)
    threading.Thread(target=_shutdown, daemon=True).start()
    return "Stopping debate and shutting down server..."


def toggle_mode_visibility(mode: str):
    """Show/hide controls based on mode. Also toggle animation area visibility."""
    single = mode == "Single Question"
    full = not single
    return (gr.update(visible=single),   # question_input
            gr.update(visible=single),   # ground_truth_input
            gr.update(visible=single),   # fetch_btn
            gr.update(visible=full),     # debate_progress (full mode only)
            gr.update(visible=True))     # animation_area (always visible)


def build_ui():
    """Build and return the Gradio interface."""
    cfg = load_config()
    debate = cfg["debate"]

    with gr.Blocks(title="LLM Debate Pipeline") as demo:
        gr.Markdown("# LLM Debate Pipeline")
        gr.Markdown("Two LLM debaters argue opposing sides. A judge renders the verdict.")

        with gr.Row():
            # --- Left column: all controls ---
            with gr.Column(scale=1):
                gr.Markdown("### Configuration")

                mode_selector = gr.Radio(
                    choices=["Single Question", "Full StrategyQA Sample"],
                    value="Single Question",
                    label="Run Mode"
                )

                question_input = gr.Textbox(
                    label="Question",
                    placeholder="Enter a yes/no question...",
                    lines=2
                )
                ground_truth_input = gr.Dropdown(
                    choices=["Yes", "No"],
                    label="Ground Truth"
                )
                fetch_btn = gr.Button("Fetch Random StrategyQA Question")

                gr.Markdown("#### Models")
                debater_a_model = gr.Dropdown(
                    choices=MODEL_CHOICES, value=debate["debater_a_model"],
                    label="Debater A Model"
                )
                debater_b_model = gr.Dropdown(
                    choices=MODEL_CHOICES, value=debate["debater_b_model"],
                    label="Debater B Model"
                )
                judge_model = gr.Dropdown(
                    choices=MODEL_CHOICES, value=debate["judge_model"],
                    label="Judge Model"
                )

                gr.Markdown("#### Temperatures")
                temp_a = gr.Slider(0.0, 1.0, value=debate["debater_a_temperature"], step=0.1,
                                   label="Debater A Temperature")
                temp_b = gr.Slider(0.0, 1.0, value=debate["debater_b_temperature"], step=0.1,
                                   label="Debater B Temperature")
                temp_judge = gr.Slider(0.0, 1.0, value=debate["judge_temperature"], step=0.1,
                                       label="Judge Temperature")

                gr.Markdown("#### Other Settings")
                max_tokens = gr.Slider(300, 800, value=debate["max_tokens"], step=100,
                                       label="Debater Max Tokens")
                judge_max_tokens = gr.Slider(300, 1500, value=debate["judge_max_tokens"], step=100,
                                             label="Judge Max Tokens")
                num_rounds = gr.Slider(3, 5, value=debate["num_rounds"], step=1,
                                       label="Number of Debate Rounds")
                num_judges = gr.Radio([1, 3], value=debate["num_judges"],
                                      label="Number of Judges")

            # --- Center column: run/stop buttons ---
            with gr.Column(scale=0, min_width=80):
                run_btn = gr.Button("RUN", elem_id="run-circle-btn")
                stop_btn = gr.Button("STOP", elem_id="stop-circle-btn")
                gr.HTML('''
                <style>
                    #run-circle-btn {
                        width: 64px !important; height: 64px !important;
                        border-radius: 50% !important; background: #22c55e !important;
                        color: white !important; font-weight: 700 !important;
                        font-size: 13px !important; border: none !important;
                        padding: 0 !important; min-width: 0 !important;
                        margin: 8px auto !important; cursor: pointer !important;
                        display: flex !important; align-items: center !important;
                        justify-content: center !important;
                    }
                    #run-circle-btn:hover { background: #16a34a !important; }
                    #stop-circle-btn {
                        width: 64px !important; height: 64px !important;
                        border-radius: 50% !important; background: #ef4444 !important;
                        color: white !important; font-weight: 700 !important;
                        font-size: 13px !important; border: none !important;
                        padding: 0 !important; min-width: 0 !important;
                        margin: 8px auto !important; cursor: pointer !important;
                        display: flex !important; align-items: center !important;
                        justify-content: center !important;
                    }
                    #stop-circle-btn:hover { background: #dc2626 !important; }
                </style>
                ''')

            # --- Right column: animation, progress ---
            with gr.Column(scale=2):
                animation_area = gr.HTML(
                    value=_svg_empty(),
                    visible=True
                )
                debate_progress = gr.Slider(
                    minimum=0, maximum=1, value=0, step=0.01,
                    label="Debate Progress",
                    interactive=False,
                    visible=False
                )

        # Event handlers
        mode_selector.change(
            fn=toggle_mode_visibility,
            inputs=mode_selector,
            outputs=[question_input, ground_truth_input, fetch_btn,
                     debate_progress, animation_area]
        )

        fetch_btn.click(
            fn=fetch_random_question,
            outputs=[question_input, ground_truth_input]
        )

        run_btn.click(
            fn=run_debate_dispatch,
            inputs=[mode_selector, question_input, ground_truth_input,
                    debater_a_model,
                    debater_b_model, judge_model, temp_a, temp_b,
                    temp_judge, max_tokens, judge_max_tokens,
                    num_rounds, num_judges],
            outputs=[debate_progress, animation_area],
        )

        stop_btn.click(
            fn=stop_and_quit,
            inputs=[],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
