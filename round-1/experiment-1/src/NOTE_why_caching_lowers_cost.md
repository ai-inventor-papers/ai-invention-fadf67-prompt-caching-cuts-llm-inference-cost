# Why prompt caching lowers LLM inference cost

Transformer inference has two distinct phases. Prefill processes the entire
prompt in parallel and is compute-bound: for an n-token prompt, attention and
MLP work grows with n² and n·d² respectively, so a 2,000-token prompt on a
7B-class model costs roughly 5×10¹² FLOPs before the first output token is
generated. Decode then produces tokens one at a time, reusing key/value (KV)
tensors for every token already processed.

Prompt caching exploits exactly this structure. Once a prompt's KV tensors
exist, they are deterministic functions of the prompt tokens alone — any later
request sharing that prefix can skip its prefill entirely and append new
tokens to the stored tensors. The incremental cost of each reused token drops
from a full n-token forward pass to a single-token append, cutting prefill
FLOPs by up to ~80% for workloads where most requests share a system prompt.
Because GPUs in the prefill phase run at high utilization and dominate
time-to-first-token and cost-per-request for long prompts, eliminating
redundant prefill translates directly into lower latency and lower compute
billed per request. The trade-off is memory: cached KV tensors occupy ~0.5 MB
per token (fp16, 32-layer 4096-dim model), so cache policy — what to store,
at what granularity, and when to evict — determines whether the FLOP savings
arrive at acceptable memory cost.
