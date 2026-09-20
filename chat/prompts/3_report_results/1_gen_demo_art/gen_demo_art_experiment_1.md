# gen_demo_art_experiment_1 — report_results

> Phase: `gen_paper_repo` · `gen_demo_art`
> Run: `run_E9LU-es1utBL` — Prompt Caching Cuts LLM Inference Cost
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `gen_demo_art_experiment_1` (sdk_openhands_agent)

### [1] SYSTEM-USER prompt · 2026-09-20 21:22:00 UTC

````
<tool_use>
Maximize parallel tool calls. Parallelize independent operations, only sequentialize dependencies.
- Multiple searches/fetches on different topics → parallel in one turn
- Search then fetch results → sequential (need URLs first)
</tool_use>

<task>
Convert this artifact's Python script into a demo notebook with MINIMAL changes to the original code.
Split into cells, add markdown explanations between sections, add a visualization cell at the end.
Output: mini_demo_data.json + code_demo.ipynb (notebook that loads data from GitHub URL)
</task>

<artifact_info>
id: art_iVpRXgHbrXUR
type: experiment
title: Simulating KV-Cache Strategies for LLM Serving Cost
summary: >-
  Pure-Python analytical simulation comparing four KV-cache reuse strategies for LLM serving (monolithic prefix caching, chunk-level
  caching, speculative approximate reuse, cache-optimal prompt restructuring) across five synthetic workloads (high sharing,
  partial overlap, similar-but-mutated prefixes, variable component order, wide-overlap), using a shared 7B-class transformer
  cost model (L=32, D=4096, H=32; prefill FLOPs = L*(n^2*d + n*4d^2); KV = 512 KiB/token fp16) with seeded LRU+TTL eviction.
  All strategies run on identical request streams under an identical cost model, so comparisons are controlled. method_out.json
  (exp_gen_sol_out schema, 55 examples; one per experiment cell with predict_monolithic / predict_chunk / predict_speculative
  / predict_restructure fields holding each strategy's full metrics) contains the main 4x4 grid (1000 requests/cell, seed
  42), an 8-point memory-budget sweep (200K-50M KV tokens), a tau x n-gram x correction-fraction sensitivity sweep for speculative
  reuse, and a wide-overlap crossover analysis. Headline findings: (1) chunk-level caching matches or beats monolithic prefix
  caching on every workload (99.8% prefill-FLOP savings on partial overlap, 99.9% on variable order) while using 36-500x less
  KV memory; (2) monolithic caching only survives when whole prefixes repeat: on a wide-overlap workload (C(12,4)=495 subset
  space) it collapses to 3.2% savings while chunk reuse stays at 99.5% - the regime that motivates chunk-granular serving
  systems; (3) n-gram (n=4,8) Jaccard matching is structurally useless for 5-15% token mutation (pairwise Jaccard ~0.3-0.5
  falls below any useful threshold), while token-set matching (n=1) at tau=0.70 recovers 74.3% prefill savings where exact
  caches give exactly 0% - a negative result about the plan's original n=8/tau=0.85 design discovered by smoke-test failure;
  (4) speculative reuse never beats exact caches when exact reuse is possible (correction-pass overhead costs ~10 points on
  high_sharing); (5) results are stable across seeds 42-45 (std of savings <= 2.2 points on every cell). Deliverables also
  include seed_robustness.json, three publication figures (PDF+PNG), a validated cost-model sanity-check script, and full/mini/preview
  variants of the schema output. Runtime ~5.7 min single-threaded on 2 CPUs.
workspace_path: >-
  /ai-inventor/aii_data/runs/run_E9LU-es1utBL/3_invention_loop/iter_1/gen_art/gen_art_experiment_1
out_expected_files:
- method.py
- full_method_out.json
- mini_method_out.json
- preview_method_out.json
out_demo_files:
- path: method.py
  description: Research methodology implementation
</artifact_info>

<github_repo>
Repo URL: https://github.com/ai-inventor-papers/ai-invention-fadf67-prompt-caching-cuts-llm-inference-cost
Raw data URL: https://raw.githubusercontent.com/ai-inventor-papers/ai-invention-fadf67-prompt-caching-cuts-llm-inference-cost/main/round-1/experiment-1/demo/mini_demo_data.json

URLs won't work yet — files pushed to GitHub AFTER notebook creation.
Use local fallback pattern so notebook works locally (now) and in Colab (after deployment).
</github_repo>

<data_file_sizes>
Data files come in three sizes:
- preview_*_out.json — READ THIS to inspect the data structure
- mini_*_out.json (~3 examples) — use for prototyping/testing
- full_*_out.json (complete) — use for the final production run. NEVER open it directly (too large to read into context). Instead, extract values programmatically with shell commands (e.g. grep) or a Python script (use aii-long-running-tasks skill for scripts).
</data_file_sizes>

<install_dependencies_pattern>
Follow the aii-colab skill exactly. It has the install cell pattern, pre-installed package list, numpy 2.0 compat shims, and all Colab-specific rules.
</install_dependencies_pattern>

<data_loading_pattern>
`mini_demo_data.json` = curated subset for the demo.
Use this pattern for Colab compatibility (GitHub URL with local fallback):
```python
GITHUB_DATA_URL = "https://raw.githubusercontent.com/ai-inventor-papers/ai-invention-fadf67-prompt-caching-cuts-llm-inference-cost/main/round-1/experiment-1/demo/mini_demo_data.json"
import json
from pathlib import Path

def load_data():
    try:
        import urllib.request
        with urllib.request.urlopen(GITHUB_DATA_URL) as response:
            return json.loads(response.read().decode())
    except Exception: pass
    local = Path("mini_demo_data.json")
    if local.exists(): return json.loads(local.read_text())
    raise FileNotFoundError("Could not load mini_demo_data.json")
```
</data_loading_pattern>

<notebook_structure>
--- Setup ---
Cell 1 (markdown): Title, description, what this artifact does.
Cell 2 (code): Install dependencies — follow the aii-colab skill's install cell pattern exactly. Fill in all packages imported by the artifact's code.
Cell 3 (code): Imports — copy original import block as-is, plus any additional imports needed for the notebook (e.g. matplotlib for visualization).
Cell 4 (code): Data loading helper — use the <data_loading_pattern> above.
Cell 5 (code): `data = load_data()`

--- Config ---
Config cell (code): Define ALL tunable parameters (iterations, epochs, n_samples, hidden_size, etc.) as variables at the top of this cell. Start with the ABSOLUTE MINIMUM values — the smallest that produce any output at all (e.g. 1 iteration, 2 samples, smallest array size). These get gradually increased during testing — see TODOs.

--- Processing ---
Remaining cells: One code cell per logical section of the original script. Add a markdown cell BEFORE each code cell. Copy code as closely as possible, with these changes:
  1. Replace file paths to use the loaded `data` variable.
  2. Use the config variables from the config cell (NOT hardcoded values).
  3. Minimal fixes are allowed if something doesn't work in notebook context (e.g. adjusting paths, removing CLI args, fixing imports), but keep changes to the absolute minimum.

--- Results ---
Visualization cell (code): Print key results in a readable table, plot numeric data with matplotlib if appropriate.
</notebook_structure>

<priority>
WORKING > OPTIMIZED. A small-scale demo that runs correctly is the goal. Once the notebook passes with minimum config values, scale up only if time permits — do NOT spend multiple retries chasing larger parameters. If a working version exists, finish and move on.
</priority>

<max_notebook_total_runtime>600s (10 min)</max_notebook_total_runtime>

<test_environment>
To test-run the notebook in a clean environment (simulating Colab), create a disposable `.nb_env` in your workspace:
```bash
/usr/local/bin/python3.12 -m venv .nb_env
.nb_env/bin/pip install -q pip jupyter ipykernel
.nb_env/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 code_demo.ipynb --output code_demo.ipynb
rm -rf .nb_env
```
The timeout is set to <max_notebook_total_runtime>. The entire notebook must finish within this time.

What happens: the .venv starts empty (just jupyter). When the notebook's install cell runs, `google.colab` is NOT in sys.modules, so ALL packages get installed — non-Colab packages unconditionally, and Colab packages (numpy, pandas, etc.) at Colab's exact versions via the guard block. The result mirrors Colab's environment as closely as possible. If a cell fails, fix the notebook and re-run.
</test_environment>

FIRST, add ALL of these to your todo list using your task/todo-tracking tool:

CRITICAL: Todo content must be copied exactly as is written here, with NO CHANGES. These todos are intentionally detailed so that another LLM could read each one without any external context and understand exactly what it has to do.


<todos>
TODO 1. Read and STRICTLY follow these skills: aii-colab, aii-long-running-tasks.
TODO 2. Read demo file and relevant preview_* files (preview only). Understand script structure: imports, setup, processing, output. Identify ALL tunable parameters (iterations, epochs, n_samples, hidden_size, batch_size, etc.) — these go in the config cell.
TODO 3. Create `mini_demo_data.json`: curated subset from at most ONE dataset (no more than 100 diverse examples). CRITICAL: do NOT read/grep full output file — may crash. Use `head -c 5000` or stream first entries with Python to pick examples.
TODO 4. Create `code_demo.ipynb` via NotebookEdit following <notebook_structure>. Set ALL config parameters to ABSOLUTE MINIMUM values — the smallest that produce any output (e.g. 1 iteration, 2 samples, smallest array sizes). Test-run using <test_environment>. Fix all errors until it passes.
TODO 5. GRADUALLY SCALE (but don't overdo it): increase config params step by step (e.g. ~2x each round). After each increase: test-run, record runtime, fix errors. STOP SCALING as soon as results look meaningful — a working small-scale demo beats a failed large-scale one. If full original params fit within <max_notebook_total_runtime> (10% margin), use them. Otherwise keep whatever works and comment out the true original values. Do NOT spend more than 2-3 scaling rounds.
TODO 6. Verify: (1) code_demo.ipynb contains GITHUB_DATA_URL = "https://raw.githubusercontent.com/ai-inventor-papers/ai-invention-fadf67-prompt-caching-cuts-llm-inference-cost/main/round-1/experiment-1/demo/mini_demo_data.json" exactly, (2) mini_demo_data.json exists, (3) uses GitHub URL pattern not just open('mini_demo_data.json').
</todos>

---

Output the result as JSON to: `/ai-inventor/aii_data/runs/run_E9LU-es1utBL/4_gen_paper_repo/_3_gen_demo_art/notebook_workspaces/iter_1/art_iVpRXgHbrXUR/.sdk_openhands_agent_struct_out.json`

JSON Schema:
```json
{
  "$defs": {
    "DemoExpectedFiles": {
      "description": "Expected output files from code demo notebook generation.",
      "properties": {
        "notebook": {
          "default": "",
          "description": "Path to the generated demo notebook. Example: 'code_demo.ipynb'",
          "title": "Notebook",
          "type": "string"
        },
        "mini_data_file": {
          "default": "",
          "description": "Path to the mini demo data JSON (curated subset). Example: 'mini_demo_data.json'",
          "title": "Mini Data File",
          "type": "string"
        }
      },
      "title": "DemoExpectedFiles",
      "type": "object"
    }
  },
  "description": "Dataset/experiment/evaluation \u2192 Jupyter notebook demo.\n\nTitle and summary come from the parent artifact (gen_art step),\nnot from the demo agent. The agent only outputs expected files.",
  "properties": {
    "out_expected_files": {
      "$ref": "#/$defs/DemoExpectedFiles",
      "description": "All output files you created. Must include the demo notebook."
    }
  },
  "required": [],
  "title": "CodeDemo",
  "type": "object"
}
```

IMPORTANT: this task is NOT complete until `/ai-inventor/aii_data/runs/run_E9LU-es1utBL/4_gen_paper_repo/_3_gen_demo_art/notebook_workspaces/iter_1/art_iVpRXgHbrXUR/.sdk_openhands_agent_struct_out.json` exists and contains JSON matching the schema above.
````

### [2] HUMAN-USER prompt · 2026-09-20 21:22:00 UTC

```
Write a 200-word note on why prompt caching lowers LLM inference cost. No experiments, no code, no literature search.
```

### [3] SKILL-INPUT — aii-colab · 2026-09-20 21:22:06 UTC

The agent loaded the **aii-colab** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

````
---
name: aii-colab
description: "Pins Google Colab's runtime for generated Jupyter notebooks — Python 3.12, the exact pre-installed versions of numpy, pandas, scikit-learn, scipy, torch, transformers and more, and the install cell guarded on google.colab that installs those versions locally but never on Colab, where reinstalling them corrupts already-loaded C extensions. ALWAYS read before writing, editing or testing any .ipynb meant to run on Colab. Triggers: Colab, notebook, ipynb, google.colab, /content, numpy.dtype size changed, ABI mismatch, np.alltrue removed in NumPy 2.0, pip install inside a notebook, nbconvert execute, Colab RAM tiers and session timeouts. NOT for plain Python scripts and repo code (aii-python), for measuring or budgeting local hardware (aii-use-hardware), or for renting cloud GPUs (aii-runpod)."
---

## Colab Runtime (as of 2026-02)

- **Python**: 3.12.12
- **OS**: Linux 6.6.105+ x86_64, glibc 2.35

## Critical Rule: Do NOT pip install pre-installed packages ON COLAB

Colab's core scientific packages have **compiled C extensions** linked against each other at specific ABI versions. Installing ANY different version (even a minor bump) partially overwrites files while the loaded `.so` extensions stay in memory, causing:

- `ValueError: numpy.dtype size changed` (numpy 1.x vs 2.x ABI)
- `ImportError: cannot import name '_center'` (numpy 2.0 vs 2.2 ABI)
- Silent corruption of scipy/sklearn/pandas internals

**On Colab: do NOT install these packages. Use Colab's versions.**
**Locally: MUST install these packages at Colab's exact versions** to match the Colab environment.

## Pre-installed Core Packages

These are pre-installed on Colab. On Colab: skip them. Locally: install at these exact versions.

```
numpy==2.0.2
pandas==2.2.2
scikit-learn==1.6.1
scipy==1.16.3
matplotlib==3.10.0
seaborn==0.13.2
torch==2.9.0+cpu
tensorflow==2.19.0
xgboost==3.1.3
lightgbm==4.6.0
networkx==3.6.1
Pillow==11.3.0
opencv-python==4.13.0.92
sympy==1.14.0
statsmodels==0.14.6
bokeh==3.7.3
plotly==5.24.1
nltk==3.9.1
spacy==3.8.11
transformers==5.0.0
datasets==4.0.0
tokenizers==0.22.2
huggingface_hub==1.4.0
openai==2.17.0
requests==2.32.4
beautifulsoup4==4.13.5
lxml==6.0.2
pydantic==2.12.3
tqdm==4.67.3
rich==13.9.4
tabulate==0.9.0
PyYAML==6.0.3
jsonschema==4.26.0
h5py==3.15.1
Cython==3.0.12
numba==0.60.0
dask==2025.12.0
polars==1.31.0
pyarrow==18.1.0
```

## Install Cell Pattern

The install cell must work on BOTH Colab and local Jupyter. Use this conditional pattern:

```python
import subprocess, sys
def _pip(*a): subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *a])

# Packages NOT pre-installed on Colab (always install everywhere)
_pip('some-rare-pkg==1.2.3')

# Core packages (pre-installed on Colab, install locally to match Colab env)
if 'google.colab' not in sys.modules:
    _pip('numpy==2.0.2', 'pandas==2.2.2', 'scikit-learn==1.6.1', 'scipy==1.16.3', 'matplotlib==3.10.0')
```

**How this works:**
- On **Colab**: `google.colab` is in `sys.modules` → skips core packages (uses Colab's pre-installed ones) → only installs non-Colab packages
- **Locally**: `google.colab` is NOT in `sys.modules` → installs core packages at Colab's exact versions → local .venv matches Colab's environment as closely as possible

Rules:
- CRITICAL: On Colab, pip installing ANY version of numpy/pandas/sklearn/scipy/matplotlib (even the same version) CORRUPTS the pre-loaded C extensions. These MUST be behind the `google.colab` guard.
- Check the pre-installed package list above. If a package is on that list, put it in the `google.colab` guard block. If not, install it unconditionally.
- For the local (non-Colab) install, use the EXACT versions from the list above so the local environment matches Colab.
- Do NOT use `--force-reinstall` — corrupts Colab system packages.
- Do NOT use `%pip` or `!pip` — use the `_pip()` helper for proper conditional control.
- `%%capture` hides install noise — only add AFTER testing is done.
- If a package requires a newer numpy/scipy than Colab has, that package is INCOMPATIBLE with Colab — find an older version or alternative.

### Example

Code imports: `numpy`, `pandas`, `sklearn`, `matplotlib`, `imodels`, `dit`, `rich`

```python
import subprocess, sys
def _pip(*a): subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *a])

# imodels, dit — NOT on Colab, always install
_pip('imodels==2.0.4')
_pip('--no-deps', 'dit==1.5')

# numpy, pandas, sklearn, matplotlib, rich — pre-installed on Colab, install locally only
if 'google.colab' not in sys.modules:
    _pip('numpy==2.0.2', 'pandas==2.2.2', 'scikit-learn==1.6.1', 'matplotlib==3.10.0', 'rich==13.9.4')
```

### Checking if a package is pre-installed

Before adding a package to the install cell, check:
1. Is it in the pre-installed list above?
2. If unsure, skip it — Colab has 500+ packages pre-installed. If the import works without installing, it's pre-installed.

## NumPy 2.0 Compatibility for Non-Colab Packages

Colab has **numpy 2.0.2**. NumPy 2.0 removed several long-deprecated APIs that older packages still use. If a non-Colab package was written for numpy 1.x, it may crash at runtime with errors like:

- `AttributeError: np.alltrue was removed in the NumPy 2.0 release`
- `AttributeError: np.sometrue was removed in the NumPy 2.0 release`
- `AttributeError: np.product was removed in the NumPy 2.0 release`

**Fix**: Add a compat shim in the imports cell (BEFORE importing the affected package):

```python
import numpy as np
if not hasattr(np, "alltrue"): np.alltrue = np.all
if not hasattr(np, "sometrue"): np.sometrue = np.any
if not hasattr(np, "product"): np.product = np.prod
```

**When to add this**: After installing non-Colab packages, test-run the notebook. If you get `AttributeError: np.X was removed`, add the corresponding shim. Common offenders: `dit`, older scientific libraries that haven't been updated for numpy 2.0.

## Colab-Specific Gotchas

1. **No kernel restart after pip install** — Unlike local Jupyter, Colab doesn't cleanly reload C extensions after pip install. Once numpy/scipy/sklearn are loaded, their C code stays in memory forever until full runtime reset.

2. **`%pip` vs `_pip()` helper** — In notebook install cells, use the `_pip()` helper (see pattern above) so you can wrap installs in Python `if` conditionals. `%pip` and `!pip` can't be used inside if/else blocks.

3. **File system** — `/content/` is the working directory. Files persist within a session but are lost on runtime disconnect.

4. **Memory** — Free tier: ~12GB RAM. Pro: ~25-50GB. Monitor with `!free -h`.

5. **Timeouts** — Free tier disconnects after ~90 min idle, ~12 hours max session. Pro: longer.

6. **`%%capture`** — Suppresses all cell output including errors. Only use on install cells AFTER verifying they work. During development, leave install output visible.

## Testing Notebooks Locally (Simulating Colab)

The goal: create a clean Python 3.12 .venv that mirrors Colab's environment, then run the notebook in it.

```bash
python3.12 -m venv .nb_env
.nb_env/bin/pip install -q pip jupyter ipykernel
.nb_env/bin/jupyter nbconvert --to notebook --execute code_demo.ipynb --output code_demo.ipynb
rm -rf .nb_env
```

**What happens when the notebook runs in this .venv:**
1. The .venv starts empty (just jupyter + ipykernel)
2. The install cell runs → `google.colab` is NOT in sys.modules
3. Non-Colab packages get installed unconditionally
4. Core packages (numpy, pandas, etc.) get installed at Colab's exact versions via the `google.colab` guard
5. Result: the .venv now has the same packages as Colab + the extra non-Colab packages = as close to Colab as possible

This ensures the notebook works from scratch with an environment matching Colab. If a cell fails, fix the notebook and re-run.
````

### [4] SYSTEM-USER prompt · 2026-09-20 21:34:09 UTC

````
YOUR PREVIOUS SESSION WAS INTERRUPTED: A single operation exceeded the 720s message timeout. Each individual operation must complete within 720s. Do NOT mock, skip, or compromise your execution — still do the real work. Try to make operations run faster if possible. If a command genuinely takes longer than 720s, split it into sequential parts that each complete within the time limit.

Last messages before interruption:
  - [agent_tool_result] terminal: Tool: terminal
Result:
.nb_env/bin/jupyter
ready
  - [agent_tool_call] terminal: Tool: terminal
{
  "command": "timeout 550 .nb_env/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=550 code_demo.ipynb --output code_demo.ipynb 2>&1 | tail -20",
  "is_input": false,
  "timeout": 590.0,
  "reset": false
}
  - [agent_tool_result] terminal: Tool: terminal
Result:
Jupyter: Interactive Computing

positional arguments:
  subcommand     the subcommand to launch

options:
  -h, --help     show this help message and exit
  --version      show the versions of core jupyter packages and exit
  --config-dir   show Jupyter config dir
  --data-dir
  - [agent_tool_call] terminal: Tool: terminal
{
  "command": ".nb_env/bin/pip install -q nbconvert nbclient && timeout 550 .nb_env/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=550 code_demo.ipynb --output code_demo.ipynb 2>&1 | tail -20",
  "is_input": false,
  "timeout": 590.0,
  "reset": false
}
  - [agent_tool_result] terminal: Tool: terminal
Result:
[notice] A new release of pip is available: 25.0.1 -> 26.2.1
[notice] To update, run: python3.12 -m pip install --upgrade pip
  File "/ai-inventor/aii_data/runs/run_E9LU-es1utBL/4_gen_paper_repo/_3_gen_demo_art/notebook_workspaces/iter_1/art_iVpRXgHbrXUR/.nb_env/lib/python3.12
  - [agent_tool_call] terminal: Tool: terminal
{
  "command": ".nb_env/bin/python -m ipykernel install --user --name python3 >/dev/null 2>&1; timeout 550 .nb_env/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=550 code_demo.ipynb --output code_demo.ipynb 2>&1 | tail -15",
  "is_input": false,
  "timeout
  - [agent_tool_result] terminal: Tool: terminal
Result:
File "/ai-inventor/aii_data/runs/run_E9LU-es1utBL/4_gen_paper_repo/_3_gen_demo_art/notebook_workspaces/iter_1/art_iVpRXgHbrXUR/.nb_env/lib/python3.12/site-packages/jupyter_client/manager.py", line 107, in wrapper
    out = await method(self, *args, **kwargs)
          ^^^^^^^^
  - [agent_tool_call] terminal: Tool: terminal
{
  "command": ".nb_env/bin/python -m ipykernel install --user --name python3 2>&1; ls ~/.local/share/jupyter/kernels 2>/dev/null; .nb_env/bin/jupyter kernelspec list 2>&1",
  "is_input": false,
  "timeout": null,
  "reset": false
}
  - [agent_tool_result] terminal: Tool: terminal
Result:
/ai-inventor/aii_data/runs/run_E9LU-es1utBL/4_gen_paper_repo/_3_gen_demo_art/notebook_workspaces/iter_1/art_iVpRXgHbrXUR/.nb_env/bin/python: No module named ipykernel
py312
[ListKernelSpecs] WARNING | Native kernel (python3) is not available
[ListKernelSpecs] WARNING | Native 
  - [agent_think]: ipykernel missing in venv? Earlier pip install -q pip jupyter ipykernel may have partially timed out. Install ipykernel.
  - [agent_tool_call] terminal: Tool: terminal
{
  "command": ".nb_env/bin/pip install -q ipykernel && .nb_env/bin/python -m ipykernel install --user --name python3 && timeout 550 .nb_env/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=550 code_demo.ipynb --output code_demo.ipynb 2>&1 | tail -15",
  "is
  - [agent_tool_result] terminal: Tool: terminal
Result:
[notice] A new release of pip is available: 25.0.1 -> 26.2.1
[notice] To update, run: python3.12 -m pip install --upgrade pip
Installed kernelspec python3 in /root/.local/share/jupyter/kernels/python3
[NbConvertApp] Converting notebook code_demo.ipynb to notebook
/ai-inventor/
  - [agent_think]: Executed. Check for errors in outputs and view key results.
  - [agent_tool_call] terminal: Tool: terminal
{
  "command": "/usr/local/bin/python - <<'EOF'\nimport json\nnb=json.load(open('code_demo.ipynb'))\nerrs=[o for c in nb['cells'] for o in c.get('outputs',[]) if o.get('output_type')=='error']\nprint('errors:',len(errs))\nfor e in errs[:3]: print(e.get('ename'),e.get('evalue'))\nfor c
  - [agent_tool_result] terminal: Tool: terminal
Result:
errors: 0
----
21:32:52|INFO   |  monolithic   saved= 63.03%  prefill_saved= 72.00%  hit= 72.00%  peak=   14.68 GB  (0.01s)

----
21:32:52|INFO   |  chunk        saved= 63.03%  prefill_saved= 72.00%  hit= 72.00%  peak=   14.68 GB  (0.01s)

----
21:32:52|INFO   |  speculative  
  - [agent_think]: 0 errors, works at min scale, runs in ~seconds. Good qualitative patterns. TODO 5: scale N_REQUESTS up — try 200 or 400, still fast. Compute time is tiny (0.01-0.14s per cell at 50). Even 1000 requests would be trivial. Let's set N_REQUESTS=1000 to match original? Speculative lookup at 1000 requests
  - [agent_response]: The demo runs cleanly with meaningful results. The compute is fast enough to scale to the original 1000 requests/cell:
  - [agent_tool_call] file_editor: Tool: file_editor
{
  "command": "str_replace",
  "path": "/ai-inventor/aii_data/runs/run_E9LU-es1utBL/4_gen_paper_repo/_3_gen_demo_art/notebook_workspaces/iter_1/art_iVpRXgHbrXUR/build_nb.py",
  "file_text": null,
  "old_str": "N_REQUESTS = 50                  # original full run: 1000 requests/cel
  - [agent_tool_result] file_editor: Tool: file_editor
Result:
The file /ai-inventor/aii_data/runs/run_E9LU-es1utBL/4_gen_paper_repo/_3_gen_demo_art/notebook_workspaces/iter_1/art_iVpRXgHbrXUR/build_nb.py has been edited. Here's the result of running `cat -n` on a snippet of /ai-inventor/aii_data/runs/run_E9LU-es1utBL/4_gen_paper_repo/
  - [agent_tool_call] terminal: Tool: terminal
{
  "command": "/usr/local/bin/python build_nb.py && time timeout 550 .nb_env/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=550 code_demo.ipynb --output code_demo.ipynb 2>&1 | tail -3",
  "is_input": false,
  "timeout": 590.0,
  "reset": false
}

CONTINUE FOLLOWING THESE INSTRUCTIONS:

<tool_use>
Maximize parallel tool calls. Parallelize independent operations, only sequentialize dependencies.
- Multiple searches/fetches on different topics → parallel in one turn
- Search then fetch results → sequential (need URLs first)
</tool_use>

<task>
Convert this artifact's Python script into a demo notebook with MINIMAL changes to the original code.
Split into cells, add markdown explanations between sections, add a visualization cell at the end.
Output: mini_demo_data.json + code_demo.ipynb (notebook that loads data from GitHub URL)
</task>

<artifact_info>
id: art_iVpRXgHbrXUR
type: experiment
title: Simulating KV-Cache Strategies for LLM Serving Cost
summary: >-
  Pure-Python analytical simulation comparing four KV-cache reuse strategies for LLM serving (monolithic prefix caching, chunk-level
  caching, speculative approximate reuse, cache-optimal prompt restructuring) across five synthetic workloads (high sharing,
  partial overlap, similar-but-mutated prefixes, variable component order, wide-overlap), using a shared 7B-class transformer
  cost model (L=32, D=4096, H=32; prefill FLOPs = L*(n^2*d + n*4d^2); KV = 512 KiB/token fp16) with seeded LRU+TTL eviction.
  All strategies run on identical request streams under an identical cost model, so comparisons are controlled. method_out.json
  (exp_gen_sol_out schema, 55 examples; one per experiment cell with predict_monolithic / predict_chunk / predict_speculative
  / predict_restructure fields holding each strategy's full metrics) contains the main 4x4 grid (1000 requests/cell, seed
  42), an 8-point memory-budget sweep (200K-50M KV tokens), a tau x n-gram x correction-fraction sensitivity sweep for speculative
  reuse, and a wide-overlap crossover analysis. Headline findings: (1) chunk-level caching matches or beats monolithic prefix
  caching on every workload (99.8% prefill-FLOP savings on partial overlap, 99.9% on variable order) while using 36-500x less
  KV memory; (2) monolithic caching only survives when whole prefixes repeat: on a wide-overlap workload (C(12,4)=495 subset
  space) it collapses to 3.2% savings while chunk reuse stays at 99.5% - the regime that motivates chunk-granular serving
  systems; (3) n-gram (n=4,8) Jaccard matching is structurally useless for 5-15% token mutation (pairwise Jaccard ~0.3-0.5
  falls below any useful threshold), while token-set matching (n=1) at tau=0.70 recovers 74.3% prefill savings where exact
  caches give exactly 0% - a negative result about the plan's original n=8/tau=0.85 design discovered by smoke-test failure;
  (4) speculative reuse never beats exact caches when exact reuse is possible (correction-pass overhead costs ~10 points on
  high_sharing); (5) results are stable across seeds 42-45 (std of savings <= 2.2 points on every cell). Deliverables also
  include seed_robustness.json, three publication figures (PDF+PNG), a validated cost-model sanity-check script, and full/mini/preview
  variants of the schema output. Runtime ~5.7 min single-threaded on 2 CPUs.
workspace_path: >-
  /ai-inventor/aii_data/runs/run_E9LU-es1utBL/3_invention_loop/iter_1/gen_art/gen_art_experiment_1
out_expected_files:
- method.py
- full_method_out.json
- mini_method_out.json
- preview_method_out.json
out_demo_files:
- path: method.py
  description: Research methodology implementation
</artifact_info>

<github_repo>
Repo URL: https://github.com/ai-inventor-papers/ai-invention-fadf67-prompt-caching-cuts-llm-inference-cost
Raw data URL: https://raw.githubusercontent.com/ai-inventor-papers/ai-invention-fadf67-prompt-caching-cuts-llm-inference-cost/main/round-1/experiment-1/demo/mini_demo_data.json

URLs won't work yet — files pushed to GitHub AFTER notebook creation.
Use local fallback pattern so notebook works locally (now) and in Colab (after deployment).
</github_repo>

<data_file_sizes>
Data files come in three sizes:
- preview_*_out.json — READ THIS to inspect the data structure
- mini_*_out.json (~3 examples) — use for prototyping/testing
- full_*_out.json (complete) — use for the final production run. NEVER open it directly (too large to read into context). Instead, extract values programmatically with shell commands (e.g. grep) or a Python script (use aii-long-running-tasks skill for scripts).
</data_file_sizes>

<install_dependencies_pattern>
Follow the aii-colab skill exactly. It has the install cell pattern, pre-installed package list, numpy 2.0 compat shims, and all Colab-specific rules.
</install_dependencies_pattern>

<data_loading_pattern>
`mini_demo_data.json` = curated subset for the demo.
Use this pattern for Colab compatibility (GitHub URL with local fallback):
```python
GITHUB_DATA_URL = "https://raw.githubusercontent.com/ai-inventor-papers/ai-invention-fadf67-prompt-caching-cuts-llm-inference-cost/main/round-1/experiment-1/demo/mini_demo_data.json"
import json
from pathlib import Path

def load_data():
    try:
        import urllib.request
        with urllib.request.urlopen(GITHUB_DATA_URL) as response:
            return json.loads(response.read().decode())
    except Exception: pass
    local = Path("mini_demo_data.json")
    if local.exists(): return json.loads(local.read_text())
    raise FileNotFoundError("Could not load mini_demo_data.json")
```
</data_loading_pattern>

<notebook_structure>
--- Setup ---
Cell 1 (markdown): Title, description, what this artifact does.
Cell 2 (code): Install dependencies — follow the aii-colab skill's install cell pattern exactly. Fill in all packages imported by the artifact's code.
Cell 3 (code): Imports — copy original import block as-is, plus any additional imports needed for the notebook (e.g. matplotlib for visualization).
Cell 4 (code): Data loading helper — use the <data_loading_pattern> above.
Cell 5 (code): `data = load_data()`

--- Config ---
Config cell (code): Define ALL tunable parameters (iterations, epochs, n_samples, hidden_size, etc.) as variables at the top of this cell. Start with the ABSOLUTE MINIMUM values — the smallest that produce any output at all (e.g. 1 iteration, 2 samples, smallest array size). These get gradually increased during testing — see TODOs.

--- Processing ---
Remaining cells: One code cell per logical section of the original script. Add a markdown cell BEFORE each code cell. Copy code as closely as possible, with these changes:
  1. Replace file paths to use the loaded `data` variable.
  2. Use the config variables from the config cell (NOT hardcoded values).
  3. Minimal fixes are allowed if something doesn't work in notebook context (e.g. adjusting paths, removing CLI args, fixing imports), but keep changes to the absolute minimum.

--- Results ---
Visualization cell (code): Print key results in a readable table, plot numeric data with matplotlib if appropriate.
</notebook_structure>

<priority>
WORKING > OPTIMIZED. A small-scale demo that runs correctly is the goal. Once the notebook passes with minimum config values, scale up only if time permits — do NOT spend multiple retries chasing larger parameters. If a working version exists, finish and move on.
</priority>

<max_notebook_total_runtime>600s (10 min)</max_notebook_total_runtime>

<test_environment>
To test-run the notebook in a clean environment (simulating Colab), create a disposable `.nb_env` in your workspace:
```bash
/usr/local/bin/python3.12 -m venv .nb_env
.nb_env/bin/pip install -q pip jupyter ipykernel
.nb_env/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 code_demo.ipynb --output code_demo.ipynb
rm -rf .nb_env
```
The timeout is set to <max_notebook_total_runtime>. The entire notebook must finish within this time.

What happens: the .venv starts empty (just jupyter). When the notebook's install cell runs, `google.colab` is NOT in sys.modules, so ALL packages get installed — non-Colab packages unconditionally, and Colab packages (numpy, pandas, etc.) at Colab's exact versions via the guard block. The result mirrors Colab's environment as closely as possible. If a cell fails, fix the notebook and re-run.
</test_environment>

FIRST, add ALL of these to your todo list using your task/todo-tracking tool:

CRITICAL: Todo content must be copied exactly as is written here, with NO CHANGES. These todos are intentionally detailed so that another LLM could read each one without any external context and understand exactly what it has to do.


<todos>
TODO 1. Read and STRICTLY follow these skills: aii-colab, aii-long-running-tasks.
TODO 2. Read demo file and relevant preview_* files (preview only). Understand script structure: imports, setup, processing, output. Identify ALL tunable parameters (iterations, epochs, n_samples, hidden_size, batch_size, etc.) — these go in the config cell.
TODO 3. Create `mini_demo_data.json`: curated subset from at most ONE dataset (no more than 100 diverse examples). CRITICAL: do NOT read/grep full output file — may crash. Use `head -c 5000` or stream first entries with Python to pick examples.
TODO 4. Create `code_demo.ipynb` via NotebookEdit following <notebook_structure>. Set ALL config parameters to ABSOLUTE MINIMUM values — the smallest that produce any output (e.g. 1 iteration, 2 samples, smallest array sizes). Test-run using <test_environment>. Fix all errors until it passes.
TODO 5. GRADUALLY SCALE (but don't overdo it): increase config params step by step (e.g. ~2x each round). After each increase: test-run, record runtime, fix errors. STOP SCALING as soon as results look meaningful — a working small-scale demo beats a failed large-scale one. If full original params fit within <max_notebook_total_runtime> (10% margin), use them. Otherwise keep whatever works and comment out the true original values. Do NOT spend more than 2-3 scaling rounds.
TODO 6. Verify: (1) code_demo.ipynb contains GITHUB_DATA_URL = "https://raw.githubusercontent.com/ai-inventor-papers/ai-invention-fadf67-prompt-caching-cuts-llm-inference-cost/main/round-1/experiment-1/demo/mini_demo_data.json" exactly, (2) mini_demo_data.json exists, (3) uses GitHub URL pattern not just open('mini_demo_data.json').
</todos>

---

Output the result as JSON to: `/ai-inventor/aii_data/runs/run_E9LU-es1utBL/4_gen_paper_repo/_3_gen_demo_art/notebook_workspaces/iter_1/art_iVpRXgHbrXUR/.sdk_openhands_agent_struct_out.json`

JSON Schema:
```json
{
  "$defs": {
    "DemoExpectedFiles": {
      "description": "Expected output files from code demo notebook generation.",
      "properties": {
        "notebook": {
          "default": "",
          "description": "Path to the generated demo notebook. Example: 'code_demo.ipynb'",
          "title": "Notebook",
          "type": "string"
        },
        "mini_data_file": {
          "default": "",
          "description": "Path to the mini demo data JSON (curated subset). Example: 'mini_demo_data.json'",
          "title": "Mini Data File",
          "type": "string"
        }
      },
      "title": "DemoExpectedFiles",
      "type": "object"
    }
  },
  "description": "Dataset/experiment/evaluation \u2192 Jupyter notebook demo.\n\nTitle and summary come from the parent artifact (gen_art step),\nnot from the demo agent. The agent only outputs expected files.",
  "properties": {
    "out_expected_files": {
      "$ref": "#/$defs/DemoExpectedFiles",
      "description": "All output files you created. Must include the demo notebook."
    }
  },
  "required": [],
  "title": "CodeDemo",
  "type": "object"
}
```

IMPORTANT: this task is NOT complete until `/ai-inventor/aii_data/runs/run_E9LU-es1utBL/4_gen_paper_repo/_3_gen_demo_art/notebook_workspaces/iter_1/art_iVpRXgHbrXUR/.sdk_openhands_agent_struct_out.json` exists and contains JSON matching the schema above.

Write a 200-word note on why prompt caching lowers LLM inference cost. No experiments, no code, no literature search.
````
