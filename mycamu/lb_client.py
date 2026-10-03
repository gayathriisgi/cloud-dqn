"""
mycamu/lb_client.py
===================
Thin HTTP client for the load balancers and backend servers.

Nothing in here invents a value: routing results, Q-values, server
metrics and health flags are whatever the real services return. If a
service cannot be reached, the result says so explicitly (``reachable:
false`` / ``error``) instead of substituting numbers.

Start-up resilience: every call has a short timeout and is wrapped in a
small bounded retry (no infinite loops), so MyCamu serves pages even
while the balancers are still starting.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import requests

from config import (BALANCER_URLS, HEALTH_TIMEOUT, REQUEST_TIMEOUT, SERVER_URLS,
                    WORKLOAD_PRESETS)

_pool = ThreadPoolExecutor(max_workers=16)
_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=128)
_session.mount("http://", _adapter)


def _get(url, timeout, retries=1):
    last = None
    for attempt in range(retries + 1):
        try:
            return _session.get(url, timeout=timeout)
        except requests.RequestException as e:
            last = e
            if attempt < retries:
                time.sleep(0.3)
    raise last


def _post(url, timeout, json_body=None, retries=1):
    last = None
    for attempt in range(retries + 1):
        try:
            return _session.post(url, json=json_body, timeout=timeout)
        except requests.RequestException as e:
            last = e
            if attempt < retries:
                time.sleep(0.3)
    raise last


# ----------------------------------------------------------------------
# Routing a real request through a balancer
# ----------------------------------------------------------------------
def route_request(algorithm: str, timeout: float = REQUEST_TIMEOUT) -> Dict:
    """Send ONE real request through the chosen balancer (GET /). The
    balancer picks a backend server, forwards the request and returns its
    decision JSON. Returns the spec'd routing fields plus everything the
    balancer reported (server_metrics, reward, ...)."""
    url = BALANCER_URLS[algorithm]
    t0 = time.perf_counter()
    try:
        r = _session.get(url + "/", timeout=timeout)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        try:
            body = r.json()
        except ValueError:
            body = {}
        return {
            "algorithm": algorithm,
            "latency_ms": round(latency_ms, 3),
            "q_values": body.get("q_values"),
            "selected_server": body.get("selected_server"),
            "status_code": r.status_code,
            "success": r.status_code == 200 and bool(body.get("success", True)),
            "backend_response_time_ms": body.get("response_time_ms"),
            "decision_time_ms": body.get("decision_time_ms"),
            "reward": body.get("reward"),
            "exploration": body.get("exploration"),
            "model_loaded": body.get("model_loaded"),
            "server_metrics": body.get("server_metrics"),
            "server_metrics_before": body.get("server_metrics_before"),
            "pheromone": body.get("pheromone"),
            "reachable": True,
        }
    except requests.RequestException as e:
        return {
            "algorithm": algorithm, "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "q_values": None, "selected_server": None, "status_code": 0, "success": False,
            "reachable": False, "error": f"{type(e).__name__}: {e}",
        }


# ----------------------------------------------------------------------
# DQN information
# ----------------------------------------------------------------------
def dqn_status() -> Dict:
    try:
        r = _get(BALANCER_URLS["dqn"] + "/status", HEALTH_TIMEOUT)
        return {"reachable": True, **r.json()}
    except Exception as e:
        return {"reachable": False, "error": str(e)}


def dqn_predict() -> Dict:
    try:
        r = _get(BALANCER_URLS["dqn"] + "/predict", REQUEST_TIMEOUT, retries=0)
        if r.status_code == 404:
            return {"available": False, "error": "DQN balancer has no /predict endpoint (older image)"}
        return {"available": True, **r.json()}
    except Exception as e:
        return {"available": False, "error": str(e)}


# ----------------------------------------------------------------------
# Backend workers
# ----------------------------------------------------------------------
def classify(m: Optional[Dict], params: Optional[Dict] = None) -> str:
    """Status label derived from REAL metrics (and the workload params the
    server itself reports via GET /load)."""
    if not m:
        return "UNREACHABLE"
    if m.get("manually_failed") or not m.get("health", False):
        return "FAILED"
    cap = max(1, int(m.get("capacity", 12)))
    lat = float(m.get("average_response_time_ms") or 0.0)
    base = float((params or {}).get("base_latency_ms") or 0.0)
    if m.get("cpu_usage", 0) >= 80 or m.get("queue_length", 0) >= cap or m.get("error_rate", 0) >= 0.1:
        return "OVERLOADED"
    if lat >= 200 or base >= 200:
        return "SLOW"
    if m.get("load_pressure", 0) >= 0.4 or lat >= 100:
        return "BUSY"
    return "HEALTHY"


def _worker(name: str, url: str) -> Dict:
    out = {"name": name, "url": url}
    try:
        m = _get(url + "/metrics", HEALTH_TIMEOUT, retries=0).json()
        out["reachable"] = True
        out["metrics"] = m
    except Exception as e:
        out.update({"reachable": False, "metrics": None, "error": str(e)})
        out["status"] = "UNREACHABLE"
        return out
    try:
        out["workload"] = _get(url + "/load", HEALTH_TIMEOUT, retries=0).json()
    except Exception:
        out["workload"] = None
    out["status"] = classify(out["metrics"], out["workload"])
    return out


def workers() -> List[Dict]:
    futs = [_pool.submit(_worker, n, u) for n, u in SERVER_URLS.items()]
    return [f.result() for f in futs]


def normalize_worker_name(name: str) -> Optional[str]:
    name = name.strip().lower()
    if name in SERVER_URLS:
        return name
    if name.isdigit() and f"server{name}" in SERVER_URLS:
        return f"server{name}"
    return None


def worker_action(name: str, action: str) -> Dict:
    """Apply a REAL condition to a backend server via its control API
    (servers/app.py): POST /load (workload preset), /fail, /recover."""
    url = SERVER_URLS[name]
    steps = []
    if action in ("slow", "overload", "medium"):
        r = _post(url + "/recover", HEALTH_TIMEOUT); steps.append(("recover", r.status_code))
        r = _post(url + "/load", REQUEST_TIMEOUT, WORKLOAD_PRESETS[action]); steps.append((f"load:{action}", r.status_code))
    elif action == "fail":
        r = _post(url + "/fail", HEALTH_TIMEOUT); steps.append(("fail", r.status_code))
    elif action in ("recover", "normal"):
        r = _post(url + "/recover", HEALTH_TIMEOUT); steps.append(("recover", r.status_code))
        r = _post(url + "/load", REQUEST_TIMEOUT, WORKLOAD_PRESETS["normal"]); steps.append(("load:normal", r.status_code))
    else:
        raise ValueError(f"unknown action {action}")
    return {"server": name, "action": action, "calls": [{"call": c, "status_code": s} for c, s in steps]}


def recover_all() -> List[Dict]:
    out = []
    for n in SERVER_URLS:
        try:
            out.append(worker_action(n, "recover"))
        except Exception as e:
            out.append({"server": n, "action": "recover", "error": str(e)})
    return out


def reset_servers() -> None:
    """POST /reset on every backend (fresh counters + default workload)."""
    for u in SERVER_URLS.values():
        try:
            _post(u + "/reset", REQUEST_TIMEOUT, retries=0)
        except Exception:
            pass


# ----------------------------------------------------------------------
# Health
# ----------------------------------------------------------------------
def _health(url: str) -> Dict:
    t0 = time.perf_counter()
    try:
        r = _get(url + "/health", HEALTH_TIMEOUT, retries=0)
        return {"ok": r.status_code == 200, "status_code": r.status_code,
                "reachable": True, "response_ms": round((time.perf_counter() - t0) * 1000, 1)}
    except Exception as e:
        return {"ok": False, "status_code": 0, "reachable": False, "error": type(e).__name__}


def health_all() -> Dict:
    targets = {"round_robin": BALANCER_URLS["round_robin"], "least_connections": BALANCER_URLS["least_connections"],
               "aco": BALANCER_URLS["aco"], "dqn": BALANCER_URLS["dqn"], **SERVER_URLS}
    futs = {k: _pool.submit(_health, u) for k, u in targets.items()}
    return {k: f.result() for k, f in futs.items()}


# ----------------------------------------------------------------------
# Metric summaries over a list of REAL request results
# ----------------------------------------------------------------------
def jain(values) -> float:
    x = np.asarray(list(values), float)
    if len(x) == 0 or (x ** 2).sum() == 0:
        return 1.0
    return float(x.sum() ** 2 / (len(x) * (x ** 2).sum()))


def summarize(results: List[Dict], wall_seconds: float, degraded: Optional[List[str]] = None) -> Dict:
    n = len(results)
    if n == 0:
        return {"total_requests": 0}
    lats = np.array([r["latency_ms"] for r in results], float)
    ok = [r for r in results if r.get("success")]
    dist = {s: 0 for s in SERVER_URLS}
    for r in results:
        if r.get("selected_server") in dist:
            dist[r["selected_server"]] += 1
    cpu_sel = []
    for r in results:
        sm, sel = r.get("server_metrics") or {}, r.get("selected_server")
        if sel and sel in sm and sm[sel].get("cpu_usage") is not None:
            cpu_sel.append(float(sm[sel]["cpu_usage"]))
    out = {
        "total_requests": n,
        "successful": len(ok),
        "failed": n - len(ok),
        "success_rate": len(ok) / n,
        "avg_latency_ms": float(lats.mean()),
        "p50_latency_ms": float(np.percentile(lats, 50)),
        "p95_latency_ms": float(np.percentile(lats, 95)),
        "p99_latency_ms": float(np.percentile(lats, 99)),
        "max_latency_ms": float(lats.max()),
        "throughput_rps": len(ok) / max(1e-6, wall_seconds),
        "wall_seconds": wall_seconds,
        "server_distribution": dist,
        "jain_fairness": jain(dist.values()),
        "avg_cpu_of_selected_server": float(np.mean(cpu_sel)) if cpu_sel else None,
    }
    if degraded:
        decided = [r for r in results if r.get("selected_server")]
        good = [r for r in decided if r["selected_server"] not in degraded]
        out["routing_accuracy"] = len(good) / len(decided) if decided else None
        out["requests_to_degraded"] = len(decided) - len(good)
        out["degraded_servers"] = degraded
    return out
