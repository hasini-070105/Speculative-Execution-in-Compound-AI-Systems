# ==============================================================================
# Speculative Execution in Compound AI Systems:
# Interactive Benchmark Runner & 300-DPI Monochrome Linear Chart Exporter
# ==============================================================================

import os
import re
import sys
import asyncio
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless backend: saves crisp PNGs cleanly without popping up inline plots
import matplotlib.pyplot as plt
import nest_asyncio
import numpy as np
from PIL import Image

from jev_primitives import (
    CalibratedSystem1Gate,
    DecisionResult,
    ExecutionTelemetry,
    GroundedLocalSystem2Service,
)
from speculative_gateway import BaselineSequentialChain, SpeculativeJevGateway

nest_asyncio.apply()

# Enforce strict black-and-white / grayscale matplotlib defaults
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "grid.color": "#777777",
    "image.cmap": "gray",
})


# ==============================================================================
# 1. Strict Black-and-White Linear Comparison PNG Graphs Saved to `output/`
# ==============================================================================

_SESSION_HISTORY: List[Tuple[str, float, float, int, int]] = []


def _make_clean_query_name(text: str, max_len: int = 36) -> str:
    """Converts a query into a clean, readable title-case filename suffix."""
    words = re.findall(r"[a-zA-Z0-9]+", text.strip())
    clean = "_".join(w.capitalize() for w in words[:6])
    return clean[:max_len].rstrip("_") or "Evaluated_Query"


def _save_pure_bw_png(fig: plt.Figure, filepath: str) -> None:
    """Saves figure and converts image to 8-bit pure black-and-white/grayscale ('L') with zero RGB color."""
    fig.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    with Image.open(filepath) as img:
        img.convert("L").save(filepath)


def save_linear_comparison_graphs(
    base: ExecutionTelemetry,
    spec: ExecutionTelemetry,
    output_dir: str = "output",
) -> List[str]:
    """
    Saves 3 distinct strict black-and-white linear comparison PNG graphs (300 DPI)
    directly into `output/`—with zero color channels and no inline plotting.
    """
    os.makedirs(output_dir, exist_ok=True)
    query_tag = _make_clean_query_name(spec.query)
    saved_files: List[str] = []

    _SESSION_HISTORY.append((
        query_tag,
        base.total_latency_sec,
        spec.total_latency_sec,
        base.tokens_consumed,
        spec.tokens_consumed,
    ))

    # --------------------------------------------------------------------------
    # Graph 1: Black-and-White Linear Pipeline Stage Progression
    # --------------------------------------------------------------------------
    stages = ["Start\n(0%)", "Stage 1\nRouting / Gate", "Stage 2\nPolicy Check", "Stage 3\nFinal Output"]
    x_stages = np.arange(len(stages))

    b_t1 = base.timeline_events[0][2] if len(base.timeline_events) > 0 else base.total_latency_sec * 0.33
    b_t2 = base.timeline_events[1][2] if len(base.timeline_events) > 1 else base.total_latency_sec * 0.62
    b_t3 = base.total_latency_sec
    base_curve = [0.0, b_t1, b_t2, b_t3]

    s_t1 = spec.gate_latency_sec
    s_t2 = s_t1
    s_t3 = spec.total_latency_sec
    spec_curve = [0.0, s_t1, s_t2, s_t3]

    speedup = base.total_latency_sec / max(0.01, spec.total_latency_sec)

    fig1, ax1 = plt.subplots(figsize=(8.5, 5.2), facecolor="white")
    ax1.set_facecolor("white")
    ax1.plot(
        x_stages, base_curve,
        linestyle="--", marker="o", markersize=8, markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.8,
        linewidth=2.3, color="#333333",
        label="Without Our Model (Sequential LLM Chain)"
    )
    ax1.plot(
        x_stages, spec_curve,
        linestyle="-", marker="s", markersize=8, markerfacecolor="black", markeredgecolor="black",
        linewidth=2.8, color="black",
        label="With Our Model (Speculative Jev Gateway)"
    )
    ax1.fill_between(
        x_stages, spec_curve, base_curve,
        facecolor="#EAEAEA", edgecolor="#777777", hatch="//", alpha=0.55,
        label=f"Latency Saved ({speedup:.1f}x Faster)"
    )

    for x_i, y_b, y_s in zip(x_stages, base_curve, spec_curve):
        if x_i > 0:
            ax1.annotate(f"{y_b:.2f}s", (x_i, y_b), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, fontweight="bold", color="black")
            ax1.annotate(f"{y_s:.2f}s", (x_i, y_s), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9, fontweight="bold", color="black")

    ax1.set_xticks(x_stages)
    ax1.set_xticklabels(stages, fontsize=10, fontweight="bold", color="black")
    ax1.set_ylabel("Cumulative Elapsed Time (Seconds)", fontsize=11, fontweight="bold", color="black")
    ax1.set_title(f"Linear Pipeline Latency Progression ({speedup:.1f}x Speedup)", fontsize=13, fontweight="bold", color="black")
    ax1.set_ylim(-0.20, max(base_curve) * 1.24)
    ax1.grid(True, linestyle=":", color="#777777", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="black", fontsize=9.5)
    fig1.tight_layout()
    p1 = os.path.join(output_dir, f"Linear_Stage_Latency_Progression__{query_tag}.png")
    _save_pure_bw_png(fig1, p1)
    saved_files.append(p1)

    # --------------------------------------------------------------------------
    # Graph 2: Black-and-White Linear Cost Scaling Across Request Volume
    # --------------------------------------------------------------------------
    volumes = np.array([100, 1000, 2500, 5000, 7500, 10000])
    base_costs = volumes * base.cost_usd
    spec_costs = volumes * spec.cost_usd
    pct_saved = ((base.cost_usd - spec.cost_usd) / max(1e-6, base.cost_usd)) * 100.0

    fig2, ax2 = plt.subplots(figsize=(8.5, 5.2), facecolor="white")
    ax2.set_facecolor("white")
    ax2.plot(
        volumes, base_costs,
        linestyle="--", marker="o", markersize=7.5, markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.8,
        linewidth=2.3, color="#333333",
        label="Without Our Model (Sequential LLM Cost)"
    )
    ax2.plot(
        volumes, spec_costs,
        linestyle="-", marker="^", markersize=8.5, markerfacecolor="black", markeredgecolor="black",
        linewidth=2.8, color="black",
        label="With Our Model (Speculative Jev Cost)"
    )
    ax2.fill_between(
        volumes, spec_costs, base_costs,
        facecolor="#EAEAEA", edgecolor="#777777", hatch="//", alpha=0.55,
        label=f"Net Cost Savings ({pct_saved:.0f}% Reduced)"
    )

    for v, cb, cs in zip(volumes[[1, 3, 5]], base_costs[[1, 3, 5]], spec_costs[[1, 3, 5]]):
        ax2.annotate(f"${cb:.2f}", (v, cb), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9.5, fontweight="bold", color="black")
        ax2.annotate(f"${cs:.2f}", (v, cs), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9.5, fontweight="bold", color="black")

    ax2.set_xlabel("Production Query Volume (Number of Requests)", fontsize=11, fontweight="bold", color="black")
    ax2.set_ylabel("Cumulative API Cost (USD $)", fontsize=11, fontweight="bold", color="black")
    ax2.set_title(f"Linear API Cost Scaling Across Query Volume ({pct_saved:.0f}% Savings)", fontsize=13, fontweight="bold", color="black")
    ax2.set_ylim(-max(base_costs) * 0.08, max(base_costs) * 1.22)
    ax2.grid(True, linestyle=":", color="#777777", alpha=0.6)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="black", fontsize=9.5)
    fig2.tight_layout()
    p2 = os.path.join(output_dir, f"Linear_Cost_Scaling_Comparison__{query_tag}.png")
    _save_pure_bw_png(fig2, p2)
    saved_files.append(p2)

    # --------------------------------------------------------------------------
    # Graph 3: Black-and-White Cumulative Latency Trend Across Evaluated Queries
    # --------------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(8.5, 5.2), facecolor="white")
    ax3.set_facecolor("white")
    if len(_SESSION_HISTORY) == 1:
        q_counts = np.arange(1, 11)
        cum_base = q_counts * base.total_latency_sec
        cum_spec = q_counts * spec.total_latency_sec
        x_vals = q_counts
        x_label = "Consecutive Queries Processed (1 to 10 Queries)"
        title_str = "Linear Cumulative Latency Growth (10-Query Projection)"
    else:
        x_vals = np.arange(1, len(_SESSION_HISTORY) + 1)
        cum_base = np.cumsum([item[1] for item in _SESSION_HISTORY])
        cum_spec = np.cumsum([item[2] for item in _SESSION_HISTORY])
        x_label = "Evaluated Queries in Current Session (Query #1 -> #N)"
        title_str = f"Cumulative Session Latency Across {len(_SESSION_HISTORY)} Evaluated Queries"

    ax3.plot(
        x_vals, cum_base,
        linestyle="--", marker="o", markersize=7.5, markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.8,
        linewidth=2.3, color="#333333",
        label="Without Our Model (Cumulative LLM Time)"
    )
    ax3.plot(
        x_vals, cum_spec,
        linestyle="-", marker="D", markersize=7.5, markerfacecolor="black", markeredgecolor="black",
        linewidth=2.8, color="black",
        label="With Our Model (Cumulative Jev Time)"
    )
    ax3.fill_between(
        x_vals, cum_spec, cum_base,
        facecolor="#EAEAEA", edgecolor="#777777", hatch="//", alpha=0.55,
        label="Total Time Saved"
    )

    ax3.annotate(f"{cum_base[-1]:.2f}s", (x_vals[-1], cum_base[-1]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9.5, fontweight="bold", color="black")
    ax3.annotate(f"{cum_spec[-1]:.2f}s", (x_vals[-1], cum_spec[-1]), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9.5, fontweight="bold", color="black")

    ax3.set_xlabel(x_label, fontsize=11, fontweight="bold", color="black")
    ax3.set_ylabel("Cumulative Wait Time (Seconds)", fontsize=11, fontweight="bold", color="black")
    ax3.set_title(title_str, fontsize=13, fontweight="bold", color="black")
    ax3.set_ylim(-max(cum_base) * 0.07, max(cum_base) * 1.22)
    ax3.grid(True, linestyle=":", color="#777777", alpha=0.6)
    ax3.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="black", fontsize=9.5)
    fig3.tight_layout()
    p3 = os.path.join(output_dir, f"Linear_Cumulative_Latency_Trend__{query_tag}.png")
    _save_pure_bw_png(fig3, p3)
    saved_files.append(p3)

    return saved_files


# ==============================================================================
# 2. Interactive Input & Head-to-Head Evaluation Runner
# ==============================================================================

_SHARED_GATE: Optional[CalibratedSystem1Gate] = None
_SHARED_LLM: Optional[GroundedLocalSystem2Service] = None


def get_engines(tau: float = 0.85) -> Tuple[CalibratedSystem1Gate, GroundedLocalSystem2Service]:
    global _SHARED_GATE, _SHARED_LLM
    if _SHARED_GATE is None:
        _SHARED_GATE = CalibratedSystem1Gate(tau=tau)
    else:
        _SHARED_GATE.tau = tau
    if _SHARED_LLM is None:
        _SHARED_LLM = GroundedLocalSystem2Service()
    return _SHARED_GATE, _SHARED_LLM


async def process_query(query: str) -> Tuple[ExecutionTelemetry, ExecutionTelemetry, DecisionResult]:
    """Evaluates the exact same query Without Our Model vs. With Our Model, prints the answer, and saves pure B&W linear PNG graphs."""
    gate, llm = get_engines(tau=0.85)

    baseline_engine = BaselineSequentialChain(llm)
    spec_engine = SpeculativeJevGateway(gate, llm)

    # 1. Run WITH our model (Speculative Jev Gateway)
    telemetry_spec, decision = await spec_engine.execute(query)

    # 2. Run WITHOUT our model (Sequential LLM Chain)
    telemetry_base = await baseline_engine.execute(query)

    # Prominent Answer Display
    print("\n" + "=" * 85)
    print(f'QUERY:  "{query}"')
    print("-" * 85)
    print(f'ANSWER: {telemetry_spec.output_text}')
    print("=" * 85)

    # Plain-English Jev Gate Breakdown
    prose_sure = int(decision.needs_prose.confidence * 100)
    prose_ans = "Yes" if decision.needs_prose.value else "No"

    print(f"\nWhat Jev Sees [{decision.backend_used}] (in {decision.gate_latency_sec:.3f} seconds):")
    print(f"• Category: {decision.category.display_name} ({int(decision.category.confidence * 100)}% sure)")
    print(f"• Needs a written explanation? {prose_ans} ({prose_sure}% sure)")
    print(f"• Urgency: {decision.urgency.label} ({decision.urgency.expected_score:.2f} out of 3)")
    print(f"\nWhich Path It Takes: {decision.routing_path.value}")
    print(f"Performance: Took {telemetry_spec.total_latency_sec:.2f} seconds total | Cost: ${telemetry_spec.cost_usd:.5f} ({telemetry_spec.tokens_consumed} LLM tokens used)")
    print(f"Why This Happened: {decision.routing_reason}")

    print("\n" + "-" * 85)
    print("WITHOUT OUR MODEL (SEQUENTIAL LLM CHAIN):")
    print("-" * 85)
    print("• Approach: Sequential Multi-Step LLM Chain (Routing -> Policy Check -> Synthesis)")
    print(f"• Performance: Took {telemetry_base.total_latency_sec:.2f} seconds total | Cost: ${telemetry_base.cost_usd:.5f} ({telemetry_base.tokens_consumed} LLM tokens used)")
    print(f"• Baseline Answer: {telemetry_base.output_text}")
    print(f"• Why: {telemetry_base.why_happened}")

    print("\n" + "-" * 85)
    print("COMPARISON SUMMARY:")
    print("-" * 85)
    speedup = telemetry_base.total_latency_sec / max(0.01, telemetry_spec.total_latency_sec)
    token_saved = max(0, telemetry_base.tokens_consumed - telemetry_spec.tokens_consumed)
    print(f"{'Metric':<28} | {'Without Our Model (Just LLM)':<30} | {'With Our Model (Jev Gateway)':<25}")
    print("-" * 85)
    print(f"{'Total Response Time':<28} | {f'{telemetry_base.total_latency_sec:.2f} s':<30} | {f'{telemetry_spec.total_latency_sec:.2f} s ({speedup:.1f}x Faster)':<25}")
    print(f"{'LLM Tokens Spent':<28} | {f'{telemetry_base.tokens_consumed} tokens':<30} | {f'{telemetry_spec.tokens_consumed} tokens ({token_saved} saved)':<25}")
    print(f"{'API Cost per 1k requests':<28} | {f'${telemetry_base.cost_usd * 1000:.2f}':<30} | {f'${telemetry_spec.cost_usd * 1000:.2f}':<25}")
    print("-" * 85)

    # Save 3 distinct strict black-and-white linear comparison PNG graphs inside `output/`
    saved_paths = save_linear_comparison_graphs(telemetry_base, telemetry_spec, output_dir="output")
    print("\nSaved pure black-and-white linear comparison PNG graphs inside 'output/' folder:")
    for path in saved_paths:
        print(f"  -> {path}")

    return telemetry_base, telemetry_spec, decision


async def interactive_session():
    """Continuously takes user input on the main thread (Colab & Terminal safe) until 'exit'."""
    get_engines(tau=0.85)
    print("=" * 85)
    print("Interactive Speculative Jev Gateway Evaluator (Live Wikipedia + Qwen2.5-1.5B)")
    print("Type any query below to see the Answer and compare Without Model (LLM) vs. With Model (Jev).")
    print("All strict black-and-white linear comparison PNG charts are saved into the 'output/' folder.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 85)

    while True:
        try:
            sys.stdout.flush()
            user_query = input("\nEnter Query (or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive session.")
            break

        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit", "q"}:
            print("Session ended. All pure black-and-white PNG charts are saved in 'output/'.")
            break

        await process_query(user_query)


def run(query: Optional[str] = None):
    """Works in both Google Colab/Jupyter and standard Python terminals."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if query:
        return loop.run_until_complete(process_query(query))
    return loop.run_until_complete(interactive_session())


if __name__ == "__main__":
    run()
