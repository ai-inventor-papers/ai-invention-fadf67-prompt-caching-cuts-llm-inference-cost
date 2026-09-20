#!/usr/bin/env python3
"""Multi-seed robustness: rerun the 4x4 grid at n=300 for seeds 42/43/44/45.

Reports mean +- std of prefill-FLOP savings per (workload, strategy) cell so
that the paper's headline numbers carry variance, not just point estimates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from loguru import logger

from method import DEFAULT_BUDGET_TOKENS, STRATEGY_NAMES, simulate
from workloads import gen_workload

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEEDS = [42, 43, 44, 45]
N_REQUESTS = 300
WORKLOADS = ["high_sharing", "partial_overlap", "similar", "variable_order"]


def main() -> None:
    agg: dict[tuple[str, str], list[float]] = {}
    hit_agg: dict[tuple[str, str], list[float]] = {}
    for seed in SEEDS:
        for kind in WORKLOADS:
            wl = gen_workload(kind, N_REQUESTS, seed=seed)
            for sname in STRATEGY_NAMES:
                r = simulate(wl, sname, DEFAULT_BUDGET_TOKENS)
                agg.setdefault((kind, sname), []).append(r["prefill_flops_saved_pct"])
                hit_agg.setdefault((kind, sname), []).append(r["hit_rate_pct"])
            del wl
        logger.info(f"seed {seed} done")

    out = []
    for kind in WORKLOADS:
        for sname in STRATEGY_NAMES:
            vals = agg[(kind, sname)]
            hits = hit_agg[(kind, sname)]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            hmean = sum(hits) / len(hits)
            hvar = sum((v - hmean) ** 2 for v in hits) / (len(hits) - 1)
            out.append({
                "workload": kind,
                "strategy": sname,
                "seeds": SEEDS,
                "n_requests_per_seed": N_REQUESTS,
                "prefill_saved_mean_pct": round(mean, 3),
                "prefill_saved_std_pct": round(var ** 0.5, 3),
                "hit_rate_mean_pct": round(hmean, 3),
                "hit_rate_std_pct": round(hvar ** 0.5, 3),
            })
            logger.info(
                f"{kind:16s} {sname:12s} prefill_saved={mean:6.2f}+-{(var)**0.5:5.2f}% "
                f"hit={hmean:6.2f}+-{(hvar)**0.5:5.2f}%"
            )
    (RESULTS / "seed_robustness.json").write_text(json.dumps(out, indent=2))
    logger.info(f"wrote {RESULTS / 'seed_robustness.json'}")


if __name__ == "__main__":
    main()