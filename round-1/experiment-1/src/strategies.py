"""Four KV-cache reuse strategies sharing one interface.

Strategy interface:
  lookup(request) -> cached_len: how many leading prefix tokens are reusable
  update(request, ...)                       # insert into cache
  evict_if_needed(now)                       # TTL + LRU under token budget
  stats: hit_rate, peak_mem_tokens, etc. collected by the simulation loop.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections import OrderedDict

from cost_model import KV_BYTES_PER_TOKEN
from workloads import Request


class CacheStrategy(ABC):
    """Base class: bounded KV-token cache with TTL + LRU eviction."""

    name: str = "base"
    approximate: bool = False   # True if reused KV is NOT bit-exact

    def __init__(self, memory_budget_tokens: int, ttl_seconds: float = 600.0):
        self.memory_budget_tokens = memory_budget_tokens
        self.ttl_seconds = ttl_seconds
        # OrderedDict key -> (n_tokens, last_access_time, insertion_time)
        self.cache: OrderedDict[object, tuple[int, float, float]] = OrderedDict()
        self.total_cached_tokens = 0
        self.peak_cached_tokens = 0
        self.evictions = 0
        self.ttl_expiries = 0
        self.requests = 0
        self.hits = 0  # requests with cached_len > 0
        self.tokens_reused = 0
        self.tokens_recomputed = 0
        # additional per-request costs charged by strategies (e.g. correction FLOPs)
        self.extra_flops = 0.0
        self.quality_penalty_sum = 0.0  # side metric for approximating strategies

    # -- bookkeeping ---------------------------------------------------------
    def _touch(self, key: object, now: float) -> None:
        n_tokens, _, ins = self.cache[key]
        self.cache[key] = (n_tokens, now, ins)
        self.cache.move_to_end(key)

    def _insert(self, key: object, n_tokens: int, now: float) -> None:
        if key in self.cache:
            self._touch(key, now)
            return
        self.cache[key] = (n_tokens, now, now)
        self.total_cached_tokens += n_tokens
        self.cache.move_to_end(key)

    def evict_if_needed(self, now: float) -> None:
        """First expire TTL entries, then LRU-evict until under budget."""
        # TTL expiry
        expired = [
            k for k, (_, last, _) in self.cache.items()
            if now - last > self.ttl_seconds
        ]
        for k in expired:
            n_tokens, _, _ = self.cache.pop(k)
            self.total_cached_tokens -= n_tokens
            self.ttl_expiries += 1
        # LRU eviction under memory budget
        while self.total_cached_tokens > self.memory_budget_tokens and self.cache:
            k, (n_tokens, _, _) = self.cache.popitem(last=False)  # LRU end
            self.total_cached_tokens -= n_tokens
            self.evictions += 1
        self.peak_cached_tokens = max(self.peak_cached_tokens, self.total_cached_tokens)

    def memory_bytes(self) -> int:
        return self.total_cached_tokens * KV_BYTES_PER_TOKEN

    def peak_memory_bytes(self) -> int:
        return self.peak_cached_tokens * KV_BYTES_PER_TOKEN

    # -- strategy API --------------------------------------------------------
    @abstractmethod
    def lookup(self, request: Request, now: float) -> tuple[int, object]:
        """Return (cached_len, key_for_update). cached_len prefix tokens reusable."""

    @abstractmethod
    def update(self, request: Request, key: object, cached_len: int, now: float) -> None:
        """Insert/refresh cache entries after serving the request."""

    def record_request(self, cached_len: int, prefix_len: int) -> None:
        self.requests += 1
        self.tokens_reused += cached_len
        self.tokens_recomputed += max(0, prefix_len - cached_len)
        if cached_len > 0:
            self.hits += 1

    def summary(self) -> dict:
        hit_rate = self.hits / self.requests if self.requests else 0.0
        return {
            "requests": self.requests,
            "hit_rate": hit_rate,
            "tokens_reused": self.tokens_reused,
            "tokens_recomputed": self.tokens_recomputed,
            "peak_cached_tokens": self.peak_cached_tokens,
            "peak_memory_bytes": self.peak_memory_bytes(),
            "peak_memory_GB": self.peak_memory_bytes() / 1e9,
            "evictions": self.evictions,
            "ttl_expiries": self.ttl_expiries,
            "extra_flops": self.extra_flops,
            "quality_penalty_sum": self.quality_penalty_sum,
            "approximate": self.approximate,
        }


# --- (a) Monolithic prefix cache (BASELINE: what vLLM/Anthropic-style APC does) ---


class MonolithicPrefixCache(CacheStrategy):
    """Reuse KV iff the ENTIRE prefix hash matches exactly."""

    name = "monolithic"

    def lookup(self, request: Request, now: float) -> tuple[int, object]:
        key = request.prefix_hash()
        if key in self.cache:
            self._touch(key, now)
            return request.prefix_len, key
        return 0, key

    def update(self, request: Request, key: object, cached_len: int, now: float) -> None:
        self._insert(key, request.prefix_len, now)


# --- (b) Chunk-level cache (chunked prefix / block-granular APC) ----------------


class ChunkCache(CacheStrategy):
    """Hash each prefix component; reuse every component found in cache.

    Reused length = sum of lengths of components whose hash is cached,
    counting only the LONGEST MATCHING PREFIX RUN of components (true for
    block-granular reuse: KV reuse is only valid for a contiguous prefix).
    """

    name = "chunk"

    def lookup(self, request: Request, now: float) -> tuple[int, object]:
        cached_len = 0
        keys_used: list[object] = []
        for h, ln in request.chunk_hashes():
            if h in self.cache:
                self._touch(h, now)
                cached_len += ln
                keys_used.append(h)
            else:
                break  # contiguous-prefix constraint: stop at first miss
        return cached_len, tuple(keys_used)

    def update(self, request: Request, key: object, cached_len: int, now: float) -> None:
        # insert all chunks (they were precomputed by lookup; recompute hashes here)
        for h, ln in request.chunk_hashes():
            self._insert(h, ln, now)


# --- (c) Speculative approximate reuse (novel / exploratory) ---------------------


class SpeculativeApproxCache(CacheStrategy):
    """Reuse KV from the nearest cached prefix with Jaccard(n-gram) > tau.

    NOTE on matching granularity: with 8-gram shingles, 5-15% token mutation
    drives prefix-pair Jaccard to ~0.3-0.5, so tau=0.85 yields ZERO reuse on
    the 'similar' workload (measured in smoke tests). Per the plan's fallback
    we therefore default to token-set (n=1) Jaccard with tau=0.70, and sweep
    n in {1,4,8} x tau in {0.50,0.70,0.85} in the sensitivity analysis.

    Cost model: correction FLOPs = CORRECTION_FRACTION * full prefill of the
    divergent delta (re-using non-exact KV requires a low-rank / partial
    correction pass over affected positions).
    """

    name = "speculative"
    approximate = True

    DEFAULT_N_GRAM_N = 1
    DEFAULT_TAU = 0.70
    CORRECTION_FRACTION = 0.15
    MAX_CANDIDATES = 64  # scan only the 64 most recently used cached prefixes

    def __init__(self, memory_budget_tokens: int, ttl_seconds: float = 600.0,
                 tau: float = DEFAULT_TAU, correction_fraction: float = 0.15,
                 ngram_n: int = DEFAULT_N_GRAM_N):
        super().__init__(memory_budget_tokens, ttl_seconds)
        self.tau = tau
        self.ngram_n = ngram_n
        self.correction_fraction = correction_fraction
        self.ngrams: dict[object, set[int]] = {}  # key -> hashed n-gram set
        self.best_similarity: float = 0.0

    def _ngram_set(self, tokens: tuple[int, ...]) -> set[int]:
        n = self.ngram_n
        if len(tokens) < n:
            return {hash(tokens) & 0xFFFFFFFF}
        return {
            hash(tokens[i:i + n]) & 0xFFFFFFFF
            for i in range(len(tokens) - n + 1)
        }

    def lookup(self, request: Request, now: float) -> tuple[int, object]:
        # exact match first (free correctness win)
        key = request.prefix_hash()
        if key in self.cache:
            self._touch(key, now)
            return request.prefix_len, key
        toks = request.prefix_hash()
        q_set = self._ngram_set(toks)
        if not q_set or not self.ngrams:
            return 0, key
        # nearest neighbour by Jaccard similarity over recent candidates
        best_key, best_j = None, 0.0
        cand_keys = list(self.ngrams.keys())[-self.MAX_CANDIDATES:]
        for cand_key in cand_keys:
            if cand_key not in self.cache:
                continue
            cand_set = self.ngrams[cand_key]
            inter = len(q_set & cand_set)
            union = len(q_set) + len(cand_set) - inter
            j = inter / union if union else 0.0
            if j > best_j:
                best_key, best_j = cand_key, j
        if best_key is not None and best_j >= self.tau:
            # never reuse more tokens than the request actually has
            cached_len = min(self.cache[best_key][0], request.prefix_len)
            self._touch(best_key, now)
            self.best_similarity = best_j
            # speculative quality penalty (side metric): scales with divergence
            self.quality_penalty_sum += (1.0 - best_j) * 100.0
            return cached_len, ("spec", best_key, key)
        return 0, key

    def update(self, request: Request, key: object, cached_len: int, now: float) -> None:
        toks = request.prefix_hash()
        h = request.prefix_hash()
        self._insert(h, request.prefix_len, now)
        self.ngrams[h] = self._ngram_set(toks)
        # drop ngram sets for evicted/expired keys
        if len(self.ngrams) > 2 * len(self.cache) + 100:
            self.ngrams = {k: v for k, v in self.ngrams.items() if k in self.cache}


# --- (d) Cache-optimal restructuring (canonical component ordering) -------------


class CacheOptimalRestructure(CacheStrategy):
    """Stable components first: canonicalize order, then chunk-cache.

    Quality penalty (side metric) charged for every request whose original
    order had to be changed; the cache itself operates on canonical order.
    """

    name = "restructure"

    PENALTY_PER_REORDER = 2.0  # % quality decrement per reordered request

    def __init__(self, memory_budget_tokens: int, ttl_seconds: float = 600.0):
        super().__init__(memory_budget_tokens, ttl_seconds)
        self.reorders = 0

    def _canonical(self, request: Request) -> list[tuple[int, ...]]:
        # stable sort components by content; deterministic canonical order
        return sorted(request.components, key=lambda c: c)

    def lookup(self, request: Request, now: float) -> tuple[int, object]:
        canon = self._canonical(request)
        cached_len = 0
        keys_used: list[object] = []
        for c in canon:
            if c in self.cache:
                self._touch(c, now)
                cached_len += len(c)
                keys_used.append(c)
            else:
                break
        return cached_len, tuple(keys_used)

    def update(self, request: Request, key: object, cached_len: int, now: float) -> None:
        canon = self._canonical(request)
        for c in canon:
            if len(c) == 0:
                continue
            self._insert(c, len(c), now)
        if request.reordered and request.prefix_len > 0:
            self.reorders += 1
            self.quality_penalty_sum += self.PENALTY_PER_REORDER


STRATEGY_CLASSES = [MonolithicPrefixCache, ChunkCache, SpeculativeApproxCache,
                    CacheOptimalRestructure]


def make_strategy(name: str, memory_budget_tokens: int, ttl_seconds: float = 600.0,
                  tau: float = SpeculativeApproxCache.DEFAULT_TAU,
                  correction_fraction: float = 0.15,
                  ngram_n: int = SpeculativeApproxCache.DEFAULT_N_GRAM_N) -> CacheStrategy:
    kwargs = dict(memory_budget_tokens=memory_budget_tokens, ttl_seconds=ttl_seconds)
    if name == "speculative":
        kwargs.update(tau=tau, correction_fraction=correction_fraction, ngram_n=ngram_n)
    cls = {s.name: s for s in STRATEGY_CLASSES}[name]
    return cls(**kwargs)