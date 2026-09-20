"""Synthetic workload generators for the KV-cache strategy simulation.

The four generators ARE the input data artifact for this experiment (no external
dataset is used). Each generator emits a list of Request objects.
"""

from __future__ import annotations


from dataclasses import dataclass, field
from typing import Callable

from loguru import logger


@dataclass
class Request:
    """One simulated LLM serving request."""

    request_id: int
    # prefix decomposed into components; concatenation(order given) = full prefix
    components: list[list[int]] = field(default_factory=list)
    gen_len: int = 0                # tokens to decode (answer length)
    arrival_time: float = 0.0       # seconds since simulation start
    # structural hints used by the restructuring strategy:
    component_semantic_ids: list[str] = field(default_factory=list)
    reordered: bool = False         # True if generator shuffled component order

    def __post_init__(self) -> None:
        # freeze components to tuples (hashable + fast hashing)
        self.components = [tuple(c) for c in self.components]
        self._full_key: tuple[int, ...] | None = None
        self._chunk_keys: list[tuple[tuple[int, ...], int]] | None = None

    @property
    def prefix_len(self) -> int:
        return sum(len(c) for c in self.components)

    def prefix_tokens(self) -> tuple[int, ...]:
        out: list[int] = []
        for c in self.components:
            out.extend(c)
        return tuple(out)

    def prefix_hash(self) -> tuple[int, ...]:
        """Full-prefix identity key (exact tuple)."""
        if self._full_key is None:
            self._full_key = self.prefix_tokens()
        return self._full_key

    def chunk_hashes(self) -> list[tuple[tuple[int, ...], int]]:
        """(key, length) per component; key IS the component tuple."""
        if self._chunk_keys is None:
            self._chunk_keys = [(c, len(c)) for c in self.components if len(c) > 0]
        return self._chunk_keys


def _tokens(rng, n: int, vocab: int = 32000) -> list[int]:
    return [rng.randrange(vocab) for _ in range(n)]


# --- Workload generator registry -------------------------------------------------


def gen_high_sharing(rng, n_requests: int) -> list[Request]:
    """(i) 80% of requests share an identical 2000-token prefix; 20% unique."""
    shared = _tokens(rng, 2000)
    reqs: list[Request] = []
    for i in range(n_requests):
        if rng.random() < 0.80:
            comps = [list(shared)]
            sids = ["shared_system"]
            reordered = False
        else:
            comps = [_tokens(rng, 2000)]
            sids = ["unique_system"]
            reordered = False
        reqs.append(
            Request(
                request_id=i,
                components=comps,
                gen_len=rng.randint(100, 500),
                component_semantic_ids=sids,
                reordered=reordered,
            )
        )
    return reqs


def gen_partial_overlap(rng, n_requests: int) -> list[Request]:
    """(ii) 5 components of ~400 tokens; each request uses a random 3-of-5 subset."""
    pool = [(f"component_{j}", _tokens(rng, 400)) for j in range(5)]
    reqs: list[Request] = []
    for i in range(n_requests):
        chosen = rng.sample(range(5), 3)
        reqs.append(
            Request(
                request_id=i,
                components=[list(pool[j][1]) for j in chosen],
                gen_len=rng.randint(100, 500),
                component_semantic_ids=[pool[j][0] for j in chosen],
            )
        )
    return reqs


def gen_similar(rng, n_requests: int, vocab: int = 32000) -> list[Request]:
    """(iii) base 2000-token prefix; each request mutates 5-15% of tokens."""
    base = _tokens(rng, 2000)
    reqs: list[Request] = []
    for i in range(n_requests):
        frac = rng.uniform(0.05, 0.15)
        n_mut = int(round(frac * 2000))
        comps = list(base)
        idxs = rng.sample(range(2000), n_mut)
        for idx in idxs:
            comps[idx] = rng.randrange(vocab)
        reqs.append(
            Request(
                request_id=i,
                components=[comps],
                gen_len=rng.randint(100, 500),
                component_semantic_ids=["mutated_system"],
            )
        )
    return reqs


def gen_variable_order(rng, n_requests: int) -> list[Request]:
    """(iv) same 5 semantic components, shuffled order per request."""
    pool = [(f"component_{j}", _tokens(rng, 400)) for j in range(5)]
    reqs: list[Request] = []
    for i in range(n_requests):
        order = list(range(5))
        rng.shuffle(order)
        is_shuffled = order != sorted(order)
        reqs.append(
            Request(
                request_id=i,
                components=[list(pool[j][1]) for j in order],
                gen_len=rng.randint(100, 500),
                component_semantic_ids=[pool[j][0] for j in order],
                reordered=is_shuffled,
            )
        )
    return reqs


def gen_partial_overlap_wide(rng, n_requests: int) -> list[Request]:
    """(v) 12 components x 400 tokens; each request uses a random 4-of-12 subset.

    Subset space C(12,4) = 495 >> request count, so full-prefix exact repeats
    are rare: this is the regime where monolithic caching should collapse while
    chunk-level reuse keeps working. Used only in the crossover analysis.
    """
    pool = [(f"component_{j}", _tokens(rng, 400)) for j in range(12)]
    reqs: list[Request] = []
    for i in range(n_requests):
        chosen = rng.sample(range(12), 4)
        reqs.append(
            Request(
                request_id=i,
                components=[list(pool[j][1]) for j in chosen],
                gen_len=rng.randint(100, 500),
                component_semantic_ids=[pool[j][0] for j in chosen],
            )
        )
    return reqs


WORKLOAD_GENERATORS: dict[str, Callable] = {
    "high_sharing": gen_high_sharing,
    "partial_overlap": gen_partial_overlap,
    "similar": gen_similar,
    "variable_order": gen_variable_order,
    "partial_overlap_wide": gen_partial_overlap_wide,  # crossover analysis only
}


def gen_workload(kind: str, n_requests: int, seed: int) -> list[Request]:
    """Generate a workload with exponential inter-arrival times (rate=1 req/s)."""
    import random

    rng = random.Random(seed)
    reqs = WORKLOAD_GENERATORS[kind](rng, n_requests)
    t = 0.0
    for r in reqs:
        t += rng.expovariate(1.0)  # rate = 1 req/s
        r.arrival_time = t
    logger.info(
        f"Workload '{kind}': {n_requests} requests, "
        f"mean prefix len {sum(r.prefix_len for r in reqs)/len(reqs):.0f} tokens, "
        f"span {t/60:.1f} min"
    )
    return reqs