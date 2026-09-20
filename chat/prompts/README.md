# Prompts

Complete, auto-generated record of **every prompt the AI Inventor system gave each agent** across this run — generated at repository-upload time so it captures all steps. For the full conversation (assistant turns, thinking, tool calls and results) see the sibling `../messages/` folder.

- Run: `run_E9LU-es1utBL` — Prompt Caching Cuts LLM Inference Cost

Each prompt is labelled by type and timestamped, with its full untruncated body:

- **SYSTEM-USER** — the pipeline-generated role/instruction prompt placed in the user slot.
- **HUMAN-USER** — the task / human-typed message into the agent stream.
- **SKILL-INPUT** — a skill the agent loaded; its `SKILL.md` instructions, verbatim.

Layout mirrors the run's module tree: one folder per high-level phase, a `round_N/` per iteration where the phase iterates, then each module — a single-task module is one `.md` file, a parallel module (gen_plan / gen_art / gen_viz / gen_demo_art) is a folder with one `.md` per task.

## Index

- **1. create_idea** — `hypo_loop`
  - round_1
    - `chat/prompts/1_create_idea/round_1/1_gen_hypo.md` — 2 prompts
    - `chat/prompts/1_create_idea/round_1/2_review_hypo.md` — 2 prompts
- **2. test_idea** — `invention_loop`
  - round_1
    - `chat/prompts/2_test_idea/round_1/1_gen_strat.md` — 2 prompts
    - `2_gen_plan/` — 3 task(s)
      - `chat/prompts/2_test_idea/round_1/2_gen_plan/gen_plan_evaluation_1.md` — 2 prompts
      - `chat/prompts/2_test_idea/round_1/2_gen_plan/gen_plan_experiment_1.md` — 2 prompts
      - `chat/prompts/2_test_idea/round_1/2_gen_plan/gen_plan_research_1.md` — 3 prompts
    - `3_gen_art/` — 3 task(s)
      - `chat/prompts/2_test_idea/round_1/3_gen_art/gen_art_evaluation_1.md` — 6 prompts
      - `chat/prompts/2_test_idea/round_1/3_gen_art/gen_art_experiment_1.md` — 15 prompts
      - `chat/prompts/2_test_idea/round_1/3_gen_art/gen_art_research_1.md` — 6 prompts
- **3. report_results** — `gen_paper_repo`
  - `1_gen_demo_art/` — 2 task(s)
    - `chat/prompts/3_report_results/1_gen_demo_art/gen_demo_art_evaluation_1.md` — 5 prompts
    - `chat/prompts/3_report_results/1_gen_demo_art/gen_demo_art_experiment_1.md` — 4 prompts
