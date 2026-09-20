#!/usr/bin/env python3
"""Render publication-quality figures from results/*.json (matplotlib)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loguru import logger

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

COLORS = {
    "monolithic": "#4053d3",
    "chunk": "#ddb310",
    "speculative": "#b51d14",
    "restructure": "#00beff",
}
LABELS = {
    "monolithic": "Monolithic prefix",
    "chunk": "Chunk-level",
    "speculative": "Speculative approx.",
    "restructure": "Cache-optimal restructure",
}
WORKLOAD_LABELS = {
    "high_sharing": "(i) High sharing",
    "partial_overlap": "(ii) Partial overlap",
    "similar": "(iii) Similar prefixes",
    "variable_order": "(iv) Variable order",
    "partial_overlap_wide": "(v) Wide overlap",
}


def fig_main() -> None:
    grid = json.loads((RESULTS / "grid.json").read_text())
    crossover = json.loads((RESULTS / "crossover_wide_overlap.json").read_text())
    rows = grid + crossover

    strategies = ["monolithic", "chunk", "speculative", "restructure"]
    workloads = ["high_sharing", "partial_overlap", "similar", "variable_order",
                 "partial_overlap_wide"]
    by = {(r["workload"], r["strategy"]): r for r in rows}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    x = range(len(workloads))
    width = 0.2

    ax = axes[0]
    for si, s in enumerate(strategies):
        vals = [by[(w, s)]["prefill_flops_saved_pct"] for w in workloads]
        ax.bar([xi + (si - 1.5) * width for xi in x], vals, width,
               label=LABELS[s], color=COLORS[s])
    ax.set_xticks(list(x))
    ax.set_xticklabels([WORKLOAD_LABELS[w] for w in workloads],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Prefill FLOPs saved (%)")
    ax.set_title("(a) Prefill-FLOP savings by workload")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    for si, s in enumerate(strategies):
        vals = [by[(w, s)]["peak_memory_GB"] for w in workloads]
        ax.bar([xi + (si - 1.5) * width for xi in x], vals, width,
               label=LABELS[s], color=COLORS[s])
    ax.set_xticks(list(x))
    ax.set_xticklabels([WORKLOAD_LABELS[w] for w in workloads],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Peak KV memory (GB, fp16)")
    ax.set_title("(b) Peak KV-cache memory by workload")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = RESULTS / f"figure_main.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None)
    logger.info(f"wrote {RESULTS/'figure_main.pdf'} and .png")


def fig_memory_sweep() -> None:
    rows = json.loads((RESULTS / "sensitivity_memory.json").read_text())
    strategies = ["monolithic", "chunk", "speculative", "restructure"]
    budgets = sorted({r["memory_budget_tokens"] for r in rows})
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    markers = {"monolithic": "o", "chunk": "s", "speculative": "^", "restructure": "D"}
    for s in strategies:
        xs, ys = [], []
        for b in budgets:
            rs = [r for r in rows
                  if r["strategy"] == s and r["memory_budget_tokens"] == b]
            if not rs:
                continue
            avg = sum(r["flops_saved_pct"] for r in rs) / len(rs)
            xs.append(b / 1e6)
            ys.append(avg)
        ax.plot(xs, ys, marker=markers[s], label=LABELS[s], color=COLORS[s])
    ax.set_xlabel("Memory budget (million KV tokens)")
    ax.set_ylabel("Mean FLOPs saved across 4 workloads (%)")
    ax.set_title("Savings vs. memory budget")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(RESULTS / f"figure_memory_sweep.{ext}",
                    dpi=200 if ext == "png" else None)
    logger.info(f"wrote {RESULTS/'figure_memory_sweep.pdf'} and .png")


def fig_spec_sensitivity() -> None:
    rows = json.loads((RESULTS / "sensitivity_spec.json").read_text())
    spec = [r for r in rows if r["strategy"] == "speculative"
            and r.get("correction_fraction") == 0.15 and r.get("tau") is not None]
    taus = sorted({r["tau"] for r in spec})
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    markers = {1: "o", 4: "s", 8: "^"}
    ngram_colors = {1: "#00beff", 4: "#ddb310", 8: "#b51d14"}
    for n in sorted({r["ngram_n"] for r in spec}):
        xs, ys = [], []
        for tau in taus:
            rs = [r for r in spec if r["ngram_n"] == n and r["tau"] == tau]
            if not rs:
                continue
            ys.append(sum(r["flops_saved_pct"] for r in rs) / len(rs))
            xs.append(tau)
        ax.plot(xs, ys, marker=markers.get(n, "o"),
                label=f"n-gram n={n}", color=ngram_colors.get(n))
    ax.set_xlabel("Jaccard threshold tau")
    ax.set_ylabel("Mean FLOPs saved (%)")
    ax.set_title("Speculative reuse: matching granularity vs. threshold\n"
                 "(workloads: similar + high_sharing, correction fraction 0.15)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(RESULTS / f"figure_spec_sensitivity.{ext}",
                    dpi=200 if ext == "png" else None)
    logger.info(f"wrote {RESULTS/'figure_spec_sensitivity.pdf'} and .png")


@logger.catch(reraise=True)
def main() -> None:
    fig_main()
    fig_memory_sweep()
    fig_spec_sensitivity()


if __name__ == "__main__":
    main()