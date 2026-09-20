# KV-Cache Strategy Simulation (gen_art_experiment_1)

Pure-Python **analytical simulation** comparing four KV-cache reuse strategies
across five synthetic workloads, reporting prefill-FLOP savings, peak KV memory,
and cache hit rates for a 7B-class model (L=32, D=4096, H=32, head_dim=128).
No GPU, no real model, no external dataset — the synthetic workload generators
are the input data artifact.

## Headline results (1000 requests/workload, budget 20M KV tokens, seed 42)

| Workload | Best strategy | Prefill FLOPs saved | Peak KV mem | Notes |
|---|---|---|---|---|
| (i) high_sharing | monolithic (=chunk=restructure) | **79.3%** | 134 GB | exact-match regime |
| (ii) partial_overlap | chunk | **99.8%** | **1.05 GB** | 36× less memory than monolithic at equal savings |
| (iii) similar | speculative (n=1, tau=0.70) | **74.3%** | 691 GB | only strategy with any reuse (87.4% hits); monolithic = 0% |
| (iv) variable_order | chunk / restructure | **99.9%** | **1.05 GB** | restructure pays 2%/reorder quality penalty |
| (v) wide_overlap (C(12,4) subsets) | chunk | **99.5%** | 2.52 GB | monolithic collapses to 3.2% — the crossover regime |

Key findings:
1. **Granularity–memory trade-off**: chunk-level caching matches or beats
   monolithic savings everywhere while using 36–500× less KV memory.
2. **Crossover**: monolithic only survives when full prefixes repeat
   (C(5,3)=10 subset space); with C(12,4)=495 it collapses to 3.2% savings
   while chunk reuse stays at 99.5%.
3. **8-gram Jaccard matching is useless for 5–15% token mutation** (Jaccard
   ~0.3–0.5 < any useful tau). Token-set (n=1) matching with tau=0.70 recovers
   74% prefill savings; tau=0.50 reaches 90% but with more speculative reuse.
   This was discovered by smoke-test failure and is a substantive result.
4. Multi-seed (42–45, n=300) std of savings ≤ 2.2 points across all cells.

## Layout

```
method.py            main experiment: sanity checks -> smoke test -> full grid ->
                     memory-budget sweep -> speculative sensitivity -> crossover
cost_model.py        FLOP + KV-memory cost model (prefill / incremental / decode)
strategies.py        4 strategies: monolithic, chunk, speculative, restructure
workloads.py         5 synthetic workload generators (the dataset artifact)
sanity_check.py      unit checks of the cost model vs. hand calculations
run_seeds.py         multi-seed robustness (4 seeds, mean±std)
make_figures.py      matplotlib figures from results/*.json
pyproject.toml       uv project (loguru, matplotlib)
results/
  grid.json          4x4 grid with per-request traces
  summary.json       ranking, best-per-workload, survival rule, crossover
  sensitivity_memory.json   budget sweep 2M/10M/20M tokens
  sensitivity_spec.json     tau x n-gram x correction-fraction sweep
  crossover_wide_overlap.json  wide-overlap regime
  seed_robustness.json      mean±std over 4 seeds
  workloads.json            workload descriptors
  figure_main.{pdf,png}     savings + memory panels
  figure_memory_sweep.{pdf,png}
  figure_spec_sensitivity.{pdf,png}
  mini_/preview_/full_method_out.json   size-optimized schema-output variants
method_out.json      exp_gen_sol_out-schema output (96 examples)
```

## How to run

```bash
uv venv .venv --python=3.12
uv pip install --python .venv/bin/python loguru matplotlib
.venv/bin/python sanity_check.py   # unit checks (~1 s)
.venv/bin/python method.py         # full pipeline (~4.5 min)
.venv/bin/python run_seeds.py      # multi-seed robustness (~1 min)
.venv/bin/python make_figures.py   # figures (~5 s)
```

All runs are seeded (seed=42 default) and deterministic.

## Restoring removed files

Two directories are marked `delete: regenerable` in `.aii/manifest.yaml`.
Neither holds any deliverable; both come back with:

```bash
# 1. .venv/ — the Python environment
uv venv .venv --python=3.12 && \
  uv pip install --python .venv/bin/python loguru==0.7.3 matplotlib==3.11.2 numpy==2.5.3

# 2. __pycache__/ — Python bytecode cache (regenerated automatically on the
#    next execution of any script; no explicit action needed)
.venv/bin/python method.py
```

## Reproducing from scratch

Delete `results/` and `method_out*.json` and rerun the four commands above;
every number in the tables regenerates bit-identically (fixed seeds).
