# Why Prompt Caching Lowers LLM Inference Cost

Transformer inference has two phases. Prefill processes every prompt token
through all layers to build the KV cache; its cost grows with sequence length
and, crucially, with the quadratic attention term (n²·d per layer), making
long prompts disproportionately expensive. Decode then generates tokens one
at a time, reusing that KV cache.

The key observation is that much of the prefill work is repeated across
requests: system prompts, few-shot examples, retrieved documents, and tool
schemas are often byte-identical or near-identical between calls. Caching the
KV tensors of those shared prefixes means subsequent requests skip prefill
for every reused token, paying only decode-time attention over the cached
keys and values. Because prefill dominates cost for long-prompt workloads,
removing redundant prefill directly cuts compute, latency (time-to-first-token),
and the energy bill that follows it.

The saving is bounded by two forces. First, hit rate: reuse only happens when
a cached prefix actually matches, so prompt structure (stable content first,
consistent formatting) and matching granularity determine how much of the
prompt is reusable. Second, memory: stored KV tensors are not free — they
occupy accelerator memory that would otherwise serve other batches, so a
cache only pays off when the FLOPs saved exceed the amortized memory cost of
holding the tensors. Effective caching is therefore a compute-memory
tradeoff: maximize shared, stable prefix reuse while keeping the stored KV
footprint small.
