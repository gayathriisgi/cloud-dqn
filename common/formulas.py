"""
common/formulas.py
===================
Shared statistical formulas used by training logs (dqn/train.py),
evaluation (testing/metrics.py, testing/compare_algorithms.py) and the
adaptive-behaviour experiment. Keeping them in one place means the
number reported during training and the number reported in the final
comparison table are guaranteed to be computed the same way.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def jain_fairness_index(x: Sequence[float]) -> float:
    """Jain's fairness index.

        J(x_1..x_n) = (sum(x_i))^2 / (n * sum(x_i^2))

    Returns a value in (0, 1]; 1.0 means every x_i is identical
    (perfectly fair allocation), 1/n means all the load is on a single
    server (maximally unfair).

    ``x_i`` here is, per this project's convention, **the number of
    requests server i successfully completed during the measurement
    window** (i.e. the per-server share of completed work). The same
    function can be applied to any other per-server load quantity
    (e.g. active_requests, load_pressure) as long as that choice is
    documented alongside the result - see README "Evaluation metrics".
    """
    x = np.asarray(list(x), dtype=np.float64)
    n = len(x)
    if n == 0:
        return 0.0
    denom = n * float(np.sum(x ** 2))
    if denom == 0:
        return 1.0  # everyone did exactly zero work -> perfectly "fair"
    return float((np.sum(x) ** 2) / denom)


def load_variance(x: Sequence[float]) -> float:
    """Population variance of a per-server load quantity (e.g.
    load_pressure or active_requests across the 4 servers). Used both
    as the imbalance penalty inside the reward function and as an
    evaluation metric."""
    x = np.asarray(list(x), dtype=np.float64)
    if len(x) == 0:
        return 0.0
    return float(np.var(x))


def load_balance_degree(x: Sequence[float]) -> float:
    """A normalized [0, 1] "balance score": 1.0 = perfectly balanced,
    0.0 = maximally imbalanced, defined as 1 - coefficient_of_variation
    clipped to [0, 1]. Coefficient of variation = std / mean."""
    x = np.asarray(list(x), dtype=np.float64)
    if len(x) == 0:
        return 1.0
    mean = float(np.mean(x))
    if mean == 0:
        return 1.0
    cv = float(np.std(x)) / mean
    return float(max(0.0, min(1.0, 1.0 - cv)))


def percentile(values: Sequence[float], p: float) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def summarize_latencies(latencies_s: Sequence[float]) -> dict:
    """Average / median / P95 / P99 response time, all in seconds and
    milliseconds, used identically by every algorithm's evaluation
    report so the comparison table (Section 9) is apples-to-apples."""
    if not latencies_s:
        return {
            "avg_s": 0.0, "median_s": 0.0, "p95_s": 0.0, "p99_s": 0.0,
            "avg_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0,
        }
    arr = np.asarray(list(latencies_s), dtype=np.float64)
    avg = float(np.mean(arr))
    median = float(np.median(arr))
    p95 = percentile(arr, 95)
    p99 = percentile(arr, 99)
    return {
        "avg_s": avg, "median_s": median, "p95_s": p95, "p99_s": p99,
        "avg_ms": avg * 1000.0, "median_ms": median * 1000.0,
        "p95_ms": p95 * 1000.0, "p99_ms": p99 * 1000.0,
    }


def moving_average(values: Sequence[float], window: int) -> List[float]:
    if not values or window <= 1:
        return list(values)
    arr = np.asarray(list(values), dtype=np.float64)
    kernel = np.ones(window) / window
    return list(np.convolve(arr, kernel, mode="same"))
