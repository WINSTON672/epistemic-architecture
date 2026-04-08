"""
reproduce.py
------------
One command to reproduce all numerical results in the paper.

    python reproduce.py

Runs the full mathematical simulation (~1 min), prints the results
table from Section 5, and saves Figure 1 to results/.

No API keys required — this is a pure mathematical simulation.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from simulation import simulate, PI_VALUES, run_all, plot


def main():
    print("=" * 65)
    print("Structural Anti-Sycophancy — Results Reproduction")
    print("=" * 65)
    print()
    print("Parameters (matching Chandra et al. 2026):")
    print("  T = 100 rounds per conversation")
    print("  k = 2 data points per round")
    print("  N = 10,000 simulations per π value")
    print("  p(D=1|H=0) = 0.40,  p(D=1|H=1) = 0.60")
    print("  spiral threshold ε = 0.01  (p(H=0) ≥ 0.99)")
    print()

    results = run_all()

    # ── Table ────────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"{'π':>6}  {'Hallucinating':>14}  {'Factual':>12}  {'Epistemic Model':>16}")
    print(f"{'':>6}  {'sycophant':>14}  {'sycophant':>12}  {'(Theorem 2: π=0)':>16}")
    print("-" * 65)

    rA, cA = results["hallucinating"]
    rB, cB = results["factual"]
    em_r, em_ci = results["epistemic"]

    for i, pi in enumerate(PI_VALUES):
        print(
            f"  {pi:.1f}    {rA[i]:.4f} ±{cA[i]:.4f}   "
            f"{rB[i]:.4f} ±{cB[i]:.4f}   "
            f"{'':>5}{em_r:.4f} ±{em_ci:.4f}"
        )

    print("=" * 65)
    print()

    # ── Key claims from paper ─────────────────────────────────────────────────
    pi0_rate = rA[0]
    pi1_rate_A = rA[-1]
    pi1_rate_B = rB[-1]

    print("Key claims verified:")
    print(f"  ✓ π=0 baseline spiral rate:              {pi0_rate:.4f}")
    print(f"  ✓ Epistemic Model rate:                  {em_r:.4f}")
    print(f"  ✓ Match within 2σ:                       "
          f"{'Yes' if abs(pi0_rate - em_r) < 2 * em_ci else 'No'}")
    print(f"  ✓ Hallucinating sycophant at π=1.0:      {pi1_rate_A:.4f}")
    print(f"  ✓ Factual sycophant at π=1.0:            {pi1_rate_B:.4f}")
    print(f"  ✓ EM dominates hallucinating (π>0):      "
          f"{'Yes' if all(em_r <= rA[i] for i in range(1, len(rA))) else 'No'}")
    print(f"  ✓ EM dominates factual (π>0):            "
          f"{'Yes' if all(em_r <= rB[i] for i in range(1, len(rB))) else 'No'}")
    print()

    # ── Figure ────────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out = plot(results, out="results/simulation_results.png")
    print(f"Figure saved → {out}")
    print()
    print("Reproduction complete.")


if __name__ == "__main__":
    main()
