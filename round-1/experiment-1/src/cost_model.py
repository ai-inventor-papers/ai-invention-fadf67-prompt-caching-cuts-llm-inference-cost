"""Analytical cost model for transformer inference (prefill / decode / KV memory).

Pure arithmetic; shared by every caching strategy in the simulation so that
strategy comparisons are never confounded by cost-model differences.
"""

from __future__ import annotations

# --- 7B-class model constants (typical: Llama-2-7B / Mistral-7B) ---
LAYERS: int = 32          # L
HIDDEN_DIM: int = 4096    # D
NUM_HEADS: int = 32       # H
HEAD_DIM: int = 128       # D_head (D = H * D_head = 4096)
FFN_EXPANSION: int = 4    # 4x hidden expansion in MLP

BYTES_PER_FP16: int = 2

# K and V each stored per layer per head per dim: 2 * L * H * D_head * 2 bytes
KV_BYTES_PER_TOKEN: int = 2 * LAYERS * NUM_HEADS * HEAD_DIM * BYTES_PER_FP16

# FLOP coefficients (attention score/apply + feed-forward)
FLOP_PREFILL_ATTN = lambda n: LAYERS * n * n * HIDDEN_DIM          # noqa: E731
FLOP_PREFILL_FFN = lambda n: LAYERS * n * FFN_EXPANSION * HIDDEN_DIM * HIDDEN_DIM  # noqa: E731


def prefill_flops(n: int) -> float:
    """FLOPs to prefill n tokens from scratch (attention + FFN terms)."""
    if n <= 0:
        return 0.0
    return FLOP_PREFILL_ATTN(n) + FLOP_PREFILL_FFN(n)


def incremental_prefill_flops(cached_len: int, total_len: int) -> float:
    """FLOPs to prefill the delta tokens given cached_len tokens already in KV cache.

    New tokens (total_len - cached_len) attend over all total_len keys; FFN runs
    per new token. Attention over already-cached keys is a cheap append (dot
    products only) which the (total - cached) * total term captures.
    """
    if total_len <= cached_len:
        return 0.0
    n_new = total_len - cached_len
    attn = LAYERS * n_new * total_len * HIDDEN_DIM
    ffn = LAYERS * n_new * FFN_EXPANSION * HIDDEN_DIM * HIDDEN_DIM
    return attn + ffn


def decode_flops_per_token(seq_len: int) -> float:
    """FLOPs to decode one token at current sequence length seq_len (with full KV cache)."""
    if seq_len < 0:
        return 0.0
    attn = LAYERS * seq_len * HIDDEN_DIM
    ffn = LAYERS * FFN_EXPANSION * HIDDEN_DIM * HIDDEN_DIM
    return attn + ffn


def decode_flops(prefill_len: int, gen_len: int) -> float:
    """FLOPs to decode gen_len tokens after a prefill_len-token prefix."""
    return sum(decode_flops_per_token(prefill_len + i) for i in range(gen_len))


def kv_memory_bytes(kv_tokens: int) -> int:
    return kv_tokens * KV_BYTES_PER_TOKEN