"""
run_benchmark.py
----------------
Measures empirical LLM sycophancy rates and compares against the
mathematical simulation from the paper.

Usage:
    # Quick run (default) — measures π only, no full simulation
    python benchmark/run_benchmark.py --models gpt-4o claude-sonnet-4-6

    # Also run full spiral simulation (costs more API calls)
    python benchmark/run_benchmark.py --models gpt-4o --full-sim

    # Use an LLM judge for classification instead of heuristic
    python benchmark/run_benchmark.py --models gpt-4o --judge gpt-4o-mini

    # Specific topics only
    python benchmark/run_benchmark.py --models gpt-4o --topics vaccines climate

    # Verbose (print each turn)
    python benchmark/run_benchmark.py --models gpt-4o --verbose

Output:
    results/benchmark_results.png   — π per model per topic vs. simulation curve
    results/benchmark_results.json  — raw numbers
    results/spiral_results.png      — spiral rates (if --full-sim)

API key env vars:
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
"""

from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark.question_bank import TOPICS
from benchmark.llm_client import get_client
from benchmark.measure_spiral import measure_pi, run_full_simulation, PiResult, SpiralResult
from simulation import simulate, PI_VALUES


# ── Plotting helpers ──────────────────────────────────────────────────────────

# Distinct colors for up to 8 models
MODEL_COLORS = [
    "#2980b9", "#c0392b", "#27ae60", "#8e44ad",
    "#e67e22", "#16a085", "#d35400", "#2c3e50",
]

TOPIC_MARKERS = {
    "vaccines":  "o",
    "climate":   "s",
    "election":  "^",
    "medicine":  "D",
    "flat_earth":"P",
    "5g":        "X",
}


def _simulation_curve() -> tuple[np.ndarray, list[float], list[float]]:
    """Return precomputed simulation curves (hallucinating + factual)."""
    rates_A, rates_B = [], []
    for pi in PI_VALUES:
        rA, _ = simulate(pi, factual=False)
        rB, _ = simulate(pi, factual=True)
        rates_A.append(rA)
        rates_B.append(rB)
    return PI_VALUES, rates_A, rates_B


def plot_pi_results(
    pi_results: list[PiResult],
    out: str = "results/benchmark_results.png",
):
    """
    Plot measured π̂ per model per topic as vertical lines on the
    simulation curves — showing where each model sits on the spiral curve.
    """
    print("\nLoading simulation curves for comparison (this may take ~1 min)...")
    pi_vals, rates_A, rates_B = _simulation_curve()

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Background: simulation curves
    ax.plot(pi_vals, rates_A, color="#c0392b", linewidth=1.5, alpha=0.6,
            label="Sim: hallucinating sycophant")
    ax.plot(pi_vals, rates_B, color="#e67e22", linewidth=1.5, alpha=0.6,
            linestyle="--", label="Sim: factual sycophant")

    # Group by model
    models = list(dict.fromkeys(r.model for r in pi_results))
    for m_idx, model in enumerate(models):
        color = MODEL_COLORS[m_idx % len(MODEL_COLORS)]
        model_results = [r for r in pi_results if r.model == model]

        # Average π across topics for a summary point
        pi_vals_model = [r.pi_hat for r in model_results]
        pi_mean  = np.mean(pi_vals_model)
        pi_std   = np.std(pi_vals_model)

        # Vertical band showing where this model sits on the curve
        ax.axvline(pi_mean, color=color, linewidth=2.0, linestyle="-",
                   label=f"{model}  (π̂={pi_mean:.2f}±{pi_std:.2f})")
        ax.axvspan(pi_mean - pi_std, pi_mean + pi_std, alpha=0.10, color=color)

        # Individual topic scatter
        for r in model_results:
            marker = TOPIC_MARKERS.get(r.topic, "o")
            ax.errorbar(r.pi_hat, 0.01 + m_idx * 0.012, xerr=r.ci_95,
                        fmt=marker, color=color, markersize=7, capsize=3, alpha=0.7)

    ax.set_xlabel("Sycophancy rate π̂ (measured from real LLM)", fontsize=12)
    ax.set_ylabel("Predicted catastrophic spiral rate\n(from mathematical simulation)", fontsize=11)
    ax.set_title(
        "Where real LLMs sit on the theoretical spiral curve\n"
        "(vertical lines = empirical π̂; background = simulation from paper)",
        fontsize=11
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.01, max(rates_A) * 1.15)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Figure saved → {out}")


def plot_per_turn(
    pi_results: list[PiResult],
    out: str = "results/benchmark_per_turn.png",
):
    """Bar chart: sycophancy rate at each escalation turn per model."""
    models  = list(dict.fromkeys(r.model for r in pi_results))
    n_turns = max(len(r.per_turn) for r in pi_results)
    x       = np.arange(n_turns)
    width   = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for m_idx, model in enumerate(models):
        color = MODEL_COLORS[m_idx % len(MODEL_COLORS)]
        # Average per_turn across topics for this model
        model_results = [r for r in pi_results if r.model == model]
        if not model_results:
            continue
        per_turn_avg = []
        for t in range(n_turns):
            vals = [r.per_turn[t] for r in model_results if t < len(r.per_turn)]
            per_turn_avg.append(np.mean(vals) if vals else 0.0)

        offset = (m_idx - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, per_turn_avg, width * 0.9, label=model, color=color, alpha=0.8)

    labels = [f"Turn {i+1}\n(conf={c:.0%})" for i, (_, c) in
              enumerate(list(TOPICS.values())[0].escalation_sequence)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Sycophancy rate", fontsize=11)
    ax.set_title("Sycophancy rate by escalation turn\n(averaged across topics)", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Figure saved → {out}")


def plot_spiral_results(
    spiral_results: list[SpiralResult],
    out: str = "results/spiral_results.png",
):
    """Compare measured LLM spiral rates against simulation baseline."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # π=0 simulation baseline
    baseline, _ = simulate(0.0, factual=False, n=50_000)
    ax.axhline(baseline, color="gray", linewidth=1.2, linestyle=":",
               label=f"π=0 baseline = {baseline:.4f}")

    # One bar per model per topic
    labels, values, errors, colors = [], [], [], []
    models = list(dict.fromkeys(r.model for r in spiral_results))
    for m_idx, model in enumerate(models):
        color = MODEL_COLORS[m_idx % len(MODEL_COLORS)]
        for r in spiral_results:
            if r.model != model:
                continue
            labels.append(f"{model}\n{r.topic}")
            values.append(r.spiral_rate)
            errors.append(r.ci_95)
            colors.append(color)

    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, alpha=0.8, yerr=errors, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Spiral rate (full T-round simulation)", fontsize=11)
    ax.set_title("Empirical LLM spiral rates vs. π=0 baseline", fontsize=11)
    ax.set_ylim(0, max(values + [baseline]) * 1.2 if values else 0.1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Figure saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark real LLM sycophancy rates")
    parser.add_argument("--models",   nargs="+",
                        default=["gpt-4o"],
                        help="Model IDs to benchmark")
    parser.add_argument("--topics",   nargs="+",
                        default=list(TOPICS.keys()),
                        help="Topic names (default: all)")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Trials per topic for π measurement (default: 50)")
    parser.add_argument("--full-sim", action="store_true",
                        help="Also run full T-round spiral simulation")
    parser.add_argument("--sim-n",    type=int, default=20,
                        help="Conversations for full sim (default: 20)")
    parser.add_argument("--sim-t",    type=int, default=20,
                        help="Rounds per conversation (default: 20, paper: 100)")
    parser.add_argument("--judge",    type=str, default=None,
                        help="Model to use as LLM judge (default: heuristic classifier)")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    # Validate topics
    unknown = [t for t in args.topics if t not in TOPICS]
    if unknown:
        print(f"Unknown topics: {unknown}. Available: {list(TOPICS.keys())}")
        sys.exit(1)

    selected_topics = {k: TOPICS[k] for k in args.topics}

    # Load judge if specified
    judge_client = None
    if args.judge:
        print(f"Loading judge model: {args.judge}")
        judge_client = get_client(args.judge)

    # ── Phase 1: Measure π ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Phase 1: Measuring empirical sycophancy rate π̂")
    print("=" * 65)

    pi_results: list[PiResult] = []

    for model_name in args.models:
        print(f"\nModel: {model_name}")
        try:
            client = get_client(model_name)
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        for topic_name, topic in selected_topics.items():
            print(f"  Topic: {topic_name} ({args.n_trials} trials × {len(topic.escalation_sequence)} turns)...")
            result = measure_pi(
                client, topic,
                n_trials    = args.n_trials,
                judge       = judge_client,
                verbose     = args.verbose,
            )
            pi_results.append(result)
            print(
                f"    π̂ = {result.pi_hat:.3f} ± {result.ci_95:.3f}  "
                f"  per-turn: {[f'{v:.2f}' for v in result.per_turn]}"
            )

    # ── Phase 1 summary ──────────────────────────────────────────────────────
    if pi_results:
        print("\n" + "=" * 65)
        print("π̂ Summary")
        print("=" * 65)
        print(f"{'Model':<35} {'Topic':<12} {'π̂':>6}  {'±95%CI':>8}")
        print("-" * 65)
        for r in pi_results:
            print(f"  {r.model:<33} {r.topic:<12} {r.pi_hat:.3f}   ±{r.ci_95:.3f}")

        # Plots
        plot_pi_results(pi_results)
        plot_per_turn(pi_results)

    # ── Phase 2: Full spiral simulation (optional) ────────────────────────────
    spiral_results: list[SpiralResult] = []

    if args.full_sim:
        print("\n" + "=" * 65)
        print(f"Phase 2: Full spiral simulation (T={args.sim_t}, N={args.sim_n})")
        print("=" * 65)

        for model_name in args.models:
            try:
                client = get_client(model_name)
            except Exception:
                continue
            for topic_name, topic in selected_topics.items():
                print(f"  {model_name} × {topic_name} ({args.sim_n} conversations × {args.sim_t} rounds)...")
                sr = run_full_simulation(
                    client, topic,
                    n_conversations = args.sim_n,
                    T               = args.sim_t,
                    judge           = judge_client,
                    verbose         = args.verbose,
                )
                spiral_results.append(sr)
                print(f"    spiral rate = {sr.spiral_rate:.3f} ± {sr.ci_95:.3f}")

        if spiral_results:
            plot_spiral_results(spiral_results)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "pi_results": [
            {
                "model":    r.model,
                "topic":    r.topic,
                "pi_hat":   round(r.pi_hat, 4),
                "ci_95":    round(r.ci_95, 4),
                "n_trials": r.n_trials,
                "per_turn": [round(v, 4) for v in r.per_turn],
            }
            for r in pi_results
        ],
        "spiral_results": [
            {
                "model":          sr.model,
                "topic":          sr.topic,
                "spiral_rate":    round(sr.spiral_rate, 4),
                "ci_95":          round(sr.ci_95, 4),
                "n_conversations":sr.n_conversations,
                "T":              sr.T,
            }
            for sr in spiral_results
        ],
    }

    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nRaw results saved → results/benchmark_results.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
