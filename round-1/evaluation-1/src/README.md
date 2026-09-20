# Evaluation: Rank Four Prompt-Caching Strategies

Statistical evaluation comparing the main hypothesis (monolithic prefix KV
caching) against three alternates (chunk-level caching, speculative
approximate reuse, cache-optimal prompt restructuring) across four synthetic
workload types, per the plan `gen_plan_evaluation_1_idx3`.

## Note on provenance
The upstream experiment artifact (`gen_art_experiment_1`) produced no
`method_out.json` at evaluation time, so `eval.py` deterministically
reproduces the simulation specified in the strategy stage (`gen_strat_1`) —
same transformer cost model (L=32, d=4096, h=32, fp16 KV), same workload
generator (1000 requests × 40 seeds per cell), same strategy definitions —
and then performs the full pre-specified evaluation on it. Every number in
`eval_out.json` is exactly reproducible by re-running the script.

## Layout
- `eval.py` — full evaluation: simulation of 4 strategies × 4 workloads
  × 40 seeds, per-cell statistics (t-based and bootstrap 95% CIs),
  pre-specified survival rule, ranking, regime flags, mechanism check.
- `eval_out.json` — primary output (`exp_eval_sol_out` schema; validated).
- `full_/mini_/preview_eval_out.json` — size-optimized variants.
- `eval_ranked_table.csv` — flat ranked comparison table.
- `logs/eval.log` — run log.

## How to run
```bash
uv venv .venv --python=3.12
uv pip install --python .venv/bin/python numpy scipy loguru
uv run --no-project python eval.py
```
Runtime ≈ 90 s on CPU.

## Key results
- Ranking (mean net savings): **restructure > chunk > speculative > monolithic**.
- **Survivor for iteration 2: `restructure`** (cache-optimal prompt
  restructuring) — ≥10% better than the monolithic baseline on 4/4 workloads
  with ~100× lower peak cache footprint.
- Monolithic prefix caching scores 0% savings under partial overlap and
  variable ordering, and 2.4% under similar prefixes (near-miss drift).
- Regime flags: no regime where the baseline suffices (alternates dominate
  all four workloads once the memory cost is charged fairly).

## Restoring removed files
- `.venv/` (deleted after the round):
  ```bash
  uv venv .venv --python=3.12 && uv pip install --python .venv/bin/python numpy scipy loguru
  ```
