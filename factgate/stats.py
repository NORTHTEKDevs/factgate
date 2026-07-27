"""Interval estimation, dependency-free.

Lives here rather than in `factgate.hallugate.policy` because that module imports rck,
which made the domain-gate benchmark -- and every documented reproduction command --
fail for anyone who had not installed a knowledge-base engine the domain gate never uses.
"""
from __future__ import annotations


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Assumes independent trials. Callers benchmarking clustered data (several trials per
    declared fact) should disclose the clustering, because this interval will be
    optimistic for them.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def cluster_wilson(clusters: dict, z: float = 1.96) -> tuple[int, int, tuple[float, float]]:
    """Wilson interval with the CLUSTER as the unit of analysis.

    `clusters` maps a cluster key to an iterable of booleans (did this trial fail?).
    A cluster counts as failing if ANY of its trials failed.

    Several trials generated from one declared fact are not independent -- they share the
    same linking behaviour on structurally similar prompts -- so a trial-level Wilson
    interval understates uncertainty. Collapsing to the fact is conservative and honest:
    it cannot claim more precision than the design supports.
    """
    keys = list(clusters)
    failed = sum(1 for k in keys if any(clusters[k]))
    return failed, len(keys), wilson(failed, len(keys), z)
