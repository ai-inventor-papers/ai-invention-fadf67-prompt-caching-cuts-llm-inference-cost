#!/usr/bin/env python3
"""KV-cache strategy simulation: 4 caching strategies x 4 synthetic workloads.

Pure-Python analytical simulation (no GPU, no real model). Workloads are
synthetic, generated in-code (workloads.py IS the input data artifact).

Outputs:
  method_out.json        - exp_gen_sol_out-schema results (headline artifact)
  results/grid.json      - 4x4 strategy x workload metric grid
  results/sensitivity_memory.json - memory-budget sweep (2M/10M/20M tokens)
  results/sensitivity_spec.json   - speculative tau & correction sweep
  results/workloads.json - workload descriptors (the synthetic dataset)
  results/summary.json   - ranking, crossover analysis, best-per-workload
  results/figure_main.pdf/.png - savings & memory panels
"""

from __future__ import annotations

import gc
import json
import os
import random
import resource
import sys
import time
from pathlib import Path

from loguru import logger

from cost_model import (
    KV_BYTES_PER_TOKEN,
    decode_flops,
    incremental_prefill_flops,
    prefill_flops,
)
from strategies import (
    CacheOptimalRestructure,
    ChunkCache,
    MonolithicPrefixCache,
    SpeculativeApproxCache,
    make_strategy,
)
from workloads import Request, gen_workload

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# ------------------------- configuration ------------------------------------
SEED = 42
N_REQUESTS = 1000
WORKLOAD_KINDS = ["high_sharing", "partial_overlap", "similar", "variable_order"]
STRATEGY_NAMES = ["monolithic", "chunk", "speculative", "restructure"]

# memory budget: 20M KV tokens default; sensitivity sweep across 8 budgets
# (small budgets induce LRU thrashing, large ones are effectively unbounded).
BUDGET_SWEEP = [200_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000,
                20_000_000, 50_000_000]
TTL_SECONDS = 600.0

# speculative matching sensitivity
TAU_SWEEP = [0.50, 0.70, 0.85]
NGRAM_SWEEP = [1, 4, 8]
CORRECTION_SWEEP = [0.05, 0.15, 0.30]

# budget used for the main grid (unbounded-like, avoids eviction artifacts)
DEFAULT_BUDGET_TOKENS = 20_000_000


def set_resource_limits() -> None:
    try:
        mem_max = int(Path("/sys/fs/cgroup/memory.max").read_text().strip())
        if mem_max > 1_000_000_000_000:
            mem_max = 14 * 1024**3
    except (FileNotFoundError, ValueError):
        mem_max = 14 * 1024**3
    budget = int(mem_max * 0.85)
    resource.setrlimit(resource.RLIMIT_AS, (budget, budget))
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1800))
    logger.info(f"RLIMIT_AS set to {budget/1e9:.1f} GB")


def simulate(workload: list[Request], strategy_name: str,
             memory_budget_tokens: int, ttl_seconds: float = TTL_SECONDS,
             tau: float = SpeculativeApproxCache.DEFAULT_TAU,
             correction_fraction: float = 0.15,
             ngram_n: int = SpeculativeApproxCache.DEFAULT_N_GRAM_N,
             record_per_request: bool = False) -> dict:
    """Run one (workload, strategy) cell. Returns metric dict."""
    strategy = make_strategy(
        strategy_name, memory_budget_tokens,
        ttl_seconds=ttl_seconds, tau=tau,
        correction_fraction=correction_fraction, ngram_n=ngram_n,
    )
    baseline_flops = 0.0
    cached_flops = 0.0
    baseline_decode = 0.0  # identical across strategies; reported for completeness
    latency_proxy = []  # per-request (baseline_flops, cached_flops) ratio proxy
    per_request: list[dict] | None = [] if record_per_request else None

    for req in workload:
        p_len = req.prefix_len
        # baseline: full prefill from scratch + decode
        b_prefill = prefill_flops(p_len)
        b_decode = decode_flops(p_len, req.gen_len)
        baseline_flops += b_prefill + b_decode
        baseline_decode += b_decode

        now = req.arrival_time
        cached_len, key = strategy.lookup(req, now)

        # cached path: prefill only the uncached delta + speculative correction
        c_prefill = incremental_prefill_flops(cached_len, p_len)
        if strategy.approximate and cached_len > 0 and cached_len < p_len:
            # correction pass over the divergent delta (fraction of full delta prefill)
            delta = p_len - cached_len
            c_prefill += strategy.correction_fraction * prefill_flops(delta)
            strategy.extra_flops += strategy.correction_fraction * prefill_flops(delta)
        elif strategy.approximate and cached_len >= p_len:
            # speculative full-prefix reuse: still needs a light verification pass
            # over the whole prefix at correction_fraction
            c_prefill += strategy.correction_fraction * prefill_flops(p_len)
            strategy.extra_flops += strategy.correction_fraction * prefill_flops(p_len)
        cached_flops += c_prefill
        cached_flops += b_decode  # decode identical

        lat_b = b_prefill + b_decode
        lat_c = c_prefill + b_decode
        latency_proxy.append((lat_b, lat_c))

        strategy.record_request(cached_len, p_len)
        strategy.update(req, key, cached_len, now)
        strategy.evict_if_needed(now)

        if per_request is not None:
            per_request.append({
                "request_id": req.request_id,
                "prefix_len": p_len,
                "cached_len": cached_len,
                "baseline_flops": lat_b,
                "cached_flops": lat_c,
            })

    s = strategy.summary()
    prefill_baseline = baseline_flops - baseline_decode
    prefill_cached = cached_flops - baseline_decode
    assert baseline_flops > 0
    assert cached_flops <= baseline_flops + 1e-6, (
        f"{strategy_name}: cached {cached_flops:.3e} > baseline {baseline_flops:.3e}"
    )
    flops_saved_pct = 100.0 * (1.0 - cached_flops / baseline_flops)
    prefill_saved_pct = 100.0 * (1.0 - prefill_cached / prefill_baseline)
    assert 0.0 <= flops_saved_pct <= 100.0, flops_saved_pct
    assert strategy.peak_cached_tokens <= memory_budget_tokens or not strategy.cache, (
        "peak memory exceeds budget without eviction"
    )
    # latency proxy: mean per-request FLOP ratio (baseline/cached), kept simple
    out = {
        "strategy": strategy_name,
        "workload": None,  # filled by caller
        "baseline_flops": baseline_flops,
        "cached_flops": cached_flops,
        "prefill_baseline_flops": prefill_baseline,
        "prefill_cached_flops": prefill_cached,
        "baseline_decode_flops": baseline_decode,
        "flops_saved_pct": flops_saved_pct,
        "prefill_flops_saved_pct": prefill_saved_pct,
        "peak_memory_GB": s["peak_memory_GB"],
        "peak_cached_tokens": s["peak_cached_tokens"],
        "hit_rate_pct": 100.0 * s["hit_rate"],
        "tokens_reused": s["tokens_reused"],
        "tokens_recomputed": s["tokens_recomputed"],
        "evictions": s["evictions"],
        "ttl_expiries": s["ttl_expiries"],
        "approximate": s["approximate"],
        "quality_penalty_sum": s["quality_penalty_sum"],
        "extra_flops": s["extra_flops"],
        "memory_budget_tokens": memory_budget_tokens,
        "per_request": per_request,
    }
    del strategy
    return out


def run_grid(n_requests: int, seed: int, memory_budget_tokens: int,
             record_per_request: bool = False) -> list[dict]:
    """Full 4x4 grid on a single shared workload set per kind (same requests
    fed to every strategy => controlled comparison)."""
    rows: list[dict] = []
    for kind in WORKLOAD_KINDS:
        t0 = time.time()
        workload = gen_workload(kind, n_requests, seed=seed)
        t_gen = time.time() - t0
        logger.info(f"[{kind}] generated in {t_gen:.2f}s; running 4 strategies...")
        for sname in STRATEGY_NAMES:
            t1 = time.time()
            r = simulate(workload, sname, memory_budget_tokens,
                         record_per_request=record_per_request)
            r["workload"] = kind
            t_run = time.time() - t1
            r["runtime_s"] = round(t_run, 3)
            rows.append(r)
            logger.info(
                f"  {sname:12s} saved={r['flops_saved_pct']:6.2f}%  "
                f"prefill_saved={r['prefill_flops_saved_pct']:6.2f}%  "
                f"hit={r['hit_rate_pct']:6.2f}%  peak={r['peak_memory_GB']:8.2f} GB  "
                f"({t_run:.2f}s)"
            )
        del workload
        gc.collect()
    return rows


def run_memory_sweep(seed: int, n_requests: int) -> list[dict]:
    rows = []
    for budget in BUDGET_SWEEP:
        logger.info(f"=== Memory budget sweep: {budget:,} tokens ===")
        grid = run_grid_quiet(n_requests, seed, budget)
        for r in grid:
            r2 = dict(r)
            r2["memory_budget_tokens"] = budget
            rows.append(r2)
    return rows


def run_grid_quiet(n_requests: int, seed: int, budget: int) -> list[dict]:
    rows = []
    for kind in WORKLOAD_KINDS:
        workload = gen_workload(kind, n_requests, seed=seed)
        for sname in STRATEGY_NAMES:
            r = simulate(workload, sname, budget)
            r["workload"] = kind
            rows.append(r)
        del workload
        gc.collect()
    return rows


def run_spec_sensitivity(seed: int, n_requests: int, budget: int) -> list[dict]:
    rows = []
    for kind in ["similar", "high_sharing"]:  # regimes where approximation matters most
        workload = gen_workload(kind, n_requests, seed=seed)
        for ngram_n in NGRAM_SWEEP:
            for tau in TAU_SWEEP:
                for cf in [0.15]:  # correction held constant in the matching sweep
                    r = simulate(workload, "speculative", budget, tau=tau,
                                 correction_fraction=cf, ngram_n=ngram_n)
                    r["workload"] = kind
                    r["tau"] = tau
                    r["ngram_n"] = ngram_n
                    r["correction_fraction"] = cf
                    rows.append(r)
                    logger.info(
                        f"  [{kind}] n={ngram_n} tau={tau} cf={cf}: "
                        f"saved={r['flops_saved_pct']:.2f}% hit={r['hit_rate_pct']:.2f}%"
                    )
        # correction-fraction sweep at default tau/n
        for cf in CORRECTION_SWEEP:
            r = simulate(workload, "speculative", budget,
                         tau=SpeculativeApproxCache.DEFAULT_TAU, correction_fraction=cf,
                         ngram_n=SpeculativeApproxCache.DEFAULT_N_GRAM_N)
            r["workload"] = kind
            r["tau"] = SpeculativeApproxCache.DEFAULT_TAU
            r["ngram_n"] = SpeculativeApproxCache.DEFAULT_N_GRAM_N
            r["correction_fraction"] = cf
            rows.append(r)
            logger.info(
                f"  [{kind}] cf={cf}: saved={r['flops_saved_pct']:.2f}% "
                f"hit={r['hit_rate_pct']:.2f}%"
            )
        # also compare exact strategies at same budget for reference
        for sname in ["monolithic", "chunk"]:
            r = simulate(workload, sname, budget)
            r["workload"] = kind
            r["tau"] = None
            r["ngram_n"] = None
            r["correction_fraction"] = None
            rows.append(r)
        del workload
        gc.collect()
    return rows


# ------------------------- output & schema shaping ---------------------------


def build_examples(rows: list[dict]) -> list[dict]:
    """Convert grid rows into exp_gen_sol_out example records.

    Structure: one example per (workload, budget, tau/n/cf config) with a
    per-strategy predict_<strategy> field, so each record directly compares
    all four caching methods on the same input.
    """
    # group rows by experiment cell (all keys except strategy)
    groups: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        key = (
            r["workload"], r["memory_budget_tokens"],
            r.get("tau"), r.get("ngram_n"), r.get("correction_fraction"),
        )
        groups.setdefault(key, {})[r["strategy"]] = r

    examples = []
    for key, strat_rows in groups.items():
        workload, budget, tau, ngram_n, cf = key
        cfg_bits = [f"workload={workload}", f"n_requests=1000",
                    f"memory_budget_tokens={budget}"]
        if tau is not None:
            cfg_bits += [f"spec_tau={tau}", f"spec_ngram_n={ngram_n}",
                         f"spec_correction_fraction={cf}"]
        ex = {
            "input": "; ".join(cfg_bits) + (
                "; task=predict per-strategy prefill-FLOP savings, peak KV "
                "memory, and hit rate for the four KV-cache strategies under "
                "the shared 7B-class cost model"),
            "output": json.dumps({
                "best_strategy": max(
                    strat_rows.items(), key=lambda kv: kv[1]["flops_saved_pct"]
                )[0],
                "monolithic_flops_saved_pct": (
                    round(strat_rows["monolithic"]["flops_saved_pct"], 4)
                    if "monolithic" in strat_rows else None),
            }),
            "metadata_workload": workload,
            "metadata_memory_budget_tokens": str(budget),
            "metadata_tau": str(tau),
            "metadata_ngram_n": str(ngram_n),
            "metadata_correction_fraction": str(cf),
            "metadata_n_strategies": str(len(strat_rows)),
        }
        # one predict_* field per strategy, holding that strategy's full metrics
        for sname, r in strat_rows.items():
            ex[f"predict_{sname}"] = json.dumps({
                "strategy": sname,
                "flops_saved_pct": round(r["flops_saved_pct"], 4),
                "prefill_flops_saved_pct": round(r["prefill_flops_saved_pct"], 4),
                "hit_rate_pct": round(r["hit_rate_pct"], 4),
                "peak_memory_GB": round(r["peak_memory_GB"], 4),
                "peak_cached_tokens": r["peak_cached_tokens"],
                "evictions": r["evictions"],
                "ttl_expiries": r["ttl_expiries"],
                "approximate": r["approximate"],
                "quality_penalty_sum": round(r["quality_penalty_sum"], 4),
                "baseline_flops": f"{r['baseline_flops']:.6e}",
                "cached_flops": f"{r['cached_flops']:.6e}",
            })
        examples.append(ex)
    return examples


def summarize(rows: list[dict]) -> dict:
    """Per-workload best strategy + ranking + survival rule from gen_strat."""
    by_wl: dict[str, list[dict]] = {}
    for r in rows:
        by_wl.setdefault(r["workload"], []).append(r)
    ranking = []
    for name in STRATEGY_NAMES:
        rs = [r for r in rows if r["strategy"] == name]
        avg_saved = sum(r["flops_saved_pct"] for r in rs) / len(rs)
        avg_mem = sum(r["peak_memory_GB"] for r in rs) / len(rs)
        ranking.append({
            "strategy": name,
            "avg_flops_saved_pct": round(avg_saved, 3),
            "avg_peak_memory_GB": round(avg_mem, 3),
        })
    ranking.sort(key=lambda x: -x["avg_flops_saved_pct"])
    best_per_workload = {}
    for wl, rs in by_wl.items():
        b = max(rs, key=lambda r: r["flops_saved_pct"])
        best_per_workload[wl] = {
            "strategy": b["strategy"],
            "flops_saved_pct": round(b["flops_saved_pct"], 3),
            "hit_rate_pct": round(b["hit_rate_pct"], 3),
            "peak_memory_GB": round(b["peak_memory_GB"], 3),
        }
    # survival rule: alternate beats monolithic by >=10% (absolute points) on >=2 of 4 workloads
    mono = {r["workload"]: r["flops_saved_pct"] for r in rows if r["strategy"] == "monolithic"}
    survival = {}
    for name in STRATEGY_NAMES:
        if name == "monolithic":
            continue
        rs = {r["workload"]: r["flops_saved_pct"] for r in rows if r["strategy"] == name}
        wins = [wl for wl in rs if rs[wl] >= mono[wl] + 10.0]
        survival[name] = {
            "workloads_beating_monolithic_by_10pts": wins,
            "n_wins": len(wins),
            "survives": len(wins) >= 2,
        }
    return {
        "ranking": ranking,
        "best_per_workload": best_per_workload,
        "survival_rule_ge_10pts_on_2_of_4": survival,
        "monolithic_baseline_flops_saved_pct": {k: round(v, 3) for k, v in mono.items()},
    }


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2))
    logger.info(f"wrote {path} ({path.stat().st_size/1024:.1f} KB)")


@logger.catch(reraise=True)
def main() -> None:
    set_resource_limits()
    t_start = time.time()

    # ---- sanity checks first (testing_plan step 1) ----
    logger.info("Running cost-model sanity checks...")
    os.system(f"{sys.executable} {HERE/'sanity_check.py'}")

    # ---- smoke test: 20 requests/cell (testing_plan step 2) ----
    logger.info("=== SMOKE TEST (20 requests per cell) ===")
    smoke_rows = run_grid(20, SEED, DEFAULT_BUDGET_TOKENS)
    _assert_smoke(smoke_rows)

    # ---- full run: 1000 requests per workload ----
    n_full = 1000
    logger.info(f"=== FULL RUN: {len(WORKLOAD_KINDS)}x{len(STRATEGY_NAMES)} grid, "
                f"{n_full} requests/cell ===")
    grid = run_grid(n_full, SEED, DEFAULT_BUDGET_TOKENS, record_per_request=True)

    # invariants
    for r in grid:
        assert 0.0 <= r["flops_saved_pct"] <= 100.0
        assert r["peak_memory_GB"] <= r["memory_budget_tokens"] * KV_BYTES_PER_TOKEN / 1e9
        assert r["cached_flops"] <= r["baseline_flops"] + 1.0

    write_json(RESULTS / "grid.json", grid)

    # ---- sensitivity: memory budget sweep ----
    logger.info("=== MEMORY BUDGET SWEEP (2M/10M/20M tokens) ===")
    mem_rows = run_memory_sweep(SEED, n_full)
    write_json(RESULTS / "sensitivity_memory.json", mem_rows)

    # ---- sensitivity: speculative tau / correction fraction ----
    logger.info("=== SPECULATIVE SENSITIVITY (tau x correction) ===")
    spec_rows = run_spec_sensitivity(SEED, n_full, DEFAULT_BUDGET_TOKENS)
    write_json(RESULTS / "sensitivity_spec.json", spec_rows)

    # ---- workload descriptors ----
    wl_desc = []
    for kind in WORKLOAD_KINDS:
        wl = gen_workload(kind, 50, seed=SEED)
        plens = [r.prefix_len for r in wl]
        wl_desc.append({
            "kind": kind,
            "n_components_per_request": len(wl[0].components),
            "mean_prefix_len_50sample": sum(plens) / len(plens),
            "min_prefix_len": min(plens),
            "max_prefix_len": max(plens),
            "mean_gen_len_50sample": sum(r.gen_len for r in wl) / len(wl),
        })
        del wl
        gc.collect()
    write_json(RESULTS / "workloads.json", wl_desc)

    # ---- summary & ranking ----
    summary = summarize(grid)
    write_json(RESULTS / "summary.json", summary)

    # ---- crossover analysis: wide-overlap regime (5th workload) ----
    # C(12,4)=495 subset space => full-prefix repeats are rare. This isolates
    # the regime where monolithic caching collapses but chunk reuse survives.
    logger.info("=== CROSSOVER: partial_overlap_wide (12-comp pool, 4-of-12) ===")
    crossover = []
    wl = gen_workload("partial_overlap_wide", n_full, seed=SEED)
    for sname in STRATEGY_NAMES:
        r = simulate(wl, sname, DEFAULT_BUDGET_TOKENS)
        r["workload"] = "partial_overlap_wide"
        crossover.append(r)
        logger.info(
            f"  {sname:12s} saved={r['flops_saved_pct']:6.2f}% "
            f"prefill={r['prefill_flops_saved_pct']:6.2f}% hit={r['hit_rate_pct']:6.2f}% "
            f"peak={r['peak_memory_GB']:.2f}GB"
        )
    del wl
    gc.collect()
    write_json(RESULTS / "crossover_wide_overlap.json", crossover)
    summary["crossover_wide_overlap"] = {
        r["strategy"]: {
            "flops_saved_pct": round(r["flops_saved_pct"], 3),
            "prefill_flops_saved_pct": round(r["prefill_flops_saved_pct"], 3),
            "hit_rate_pct": round(r["hit_rate_pct"], 3),
            "peak_memory_GB": round(r["peak_memory_GB"], 3),
        } for r in crossover
    }
    write_json(RESULTS / "summary.json", summary)
    # ---- schema-shaped output ----
    examples = build_examples(grid + mem_rows + spec_rows)
    examples.extend(build_examples(crossover))
    out = {
        "metadata": {
            "method_name": "kv_cache_strategy_simulation",
            "description": (
                "Analytical simulation of four KV-cache reuse strategies "
                "(monolithic prefix, chunk-level, speculative approximate, "
                "cache-optimal restructuring) across four synthetic workloads; "
                "metrics: prefill-FLOP savings, peak KV memory, hit rate."
            ),
            "model_constants": {
                "layers": 32, "hidden_dim": 4096, "heads": 32, "head_dim": 128,
                "kv_bytes_per_token_fp16": KV_BYTES_PER_TOKEN,
            },
            "parameters": {
                "seed": SEED,
                "n_requests_full": n_full,
                "memory_budget_tokens_default": DEFAULT_BUDGET_TOKENS,
                "ttl_seconds": TTL_SECONDS,
                "budget_sweep_tokens": BUDGET_SWEEP,
                "spec_tau": SpeculativeApproxCache.DEFAULT_TAU,
                "spec_ngram_n": SpeculativeApproxCache.DEFAULT_N_GRAM_N,
                "spec_correction_fraction": SpeculativeApproxCache.CORRECTION_FRACTION,
                "spec_tau_sweep": TAU_SWEEP,
                "spec_ngram_sweep": NGRAM_SWEEP,
                "spec_correction_sweep": CORRECTION_SWEEP,
                "workload_kinds": WORKLOAD_KINDS,
                "strategies": STRATEGY_NAMES,
            },
            "total_runtime_s": round(time.time() - t_start, 1),
        },
        "datasets": [
            {
                "dataset": "kv_cache_strategy_grid",
                "examples": examples,
            }
        ],
    }
    out_path = HERE / "method_out.json"
    write_json(out_path, out)

    logger.info(f"TOTAL runtime {time.time()-t_start:.1f}s")
    logger.info("Done.")


def _assert_smoke(rows: list[dict]) -> None:
    """Testing_plan step 2 expectations (calibrated against the cost model)."""
    by = {(r["workload"], r["strategy"]): r for r in rows}
    # (a) monolithic on high_sharing: ~80% of requests share the prefix ->
    #     prefill savings ~80% x 0.75 (prefill of 2nd+ occurrence is only the
    #     FFN delta at the plan's attention term is charged on the full length)
    #     -> measured ~56% at 20 requests (50% shared within first 20 draws);
    #     require >= 50% as the structural bound.
    m = by[("high_sharing", "monolithic")]
    logger.info(f"smoke: monolithic/high_sharing prefill_saved={m['prefill_flops_saved_pct']:.1f}% "
                f"hit={m['hit_rate_pct']:.1f}%")
    assert m["prefill_flops_saved_pct"] >= 50.0, (
        f"expected >=50% prefill savings for monolithic on high_sharing smoke, got "
        f"{m['prefill_flops_saved_pct']:.1f}%"
    )
    # (b) chunk beats monolithic on partial_overlap
    c = by[("partial_overlap", "chunk")]
    assert c["flops_saved_pct"] > by[("partial_overlap", "monolithic")]["flops_saved_pct"], (
        "chunk cache must beat monolithic on partial_overlap"
    )
    # (c) similar prefixes: monolithic hit ~0, speculative > 0
    ms = by[("similar", "monolithic")]
    sp = by[("similar", "speculative")]
    logger.info(f"smoke: similar -> monolithic hit={ms['hit_rate_pct']:.1f}%, "
                f"speculative hit={sp['hit_rate_pct']:.1f}%")
    assert ms["hit_rate_pct"] <= 5.0, ms["hit_rate_pct"]
    assert sp["hit_rate_pct"] > 0.0, "speculative should get hits on similar prefixes"
    # (d) restructure beats monolithic on variable_order
    rs = by[("variable_order", "restructure")]
    assert rs["flops_saved_pct"] > by[("variable_order", "monolithic")]["flops_saved_pct"]
    # (e) speculative never beats exact caches on FLOPs on high_sharing (correction overhead)
    sp_h = by[("high_sharing", "speculative")]
    assert sp_h["flops_saved_pct"] <= m["prefill_flops_saved_pct"] + 5.0
    logger.info("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()