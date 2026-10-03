"""
mycamu/traffic.py
=================
Traffic Control Center + Algorithm Comparison engines.

TrafficJob
  Sends N REAL requests through the selected balancer (paced at a target
  rate, bounded concurrency). Every completed request becomes an EVENT
  carrying exactly what the balancer returned (selected server, Q-values,
  latency, success). The UI animates these events - it never generates
  its own routing results. Only one job runs at a time (START / STOP /
  RESET), so duplicate traffic generators are impossible.

ComparisonJob
  Runs the same scenario + same request count for each selected
  algorithm one after another, resetting the backends and re-applying the
  scenario before each run, and stores the measured summaries.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from config import ALGORITHM_LABELS, BALANCER_URLS, MAX_TRAFFIC_REQUESTS, SERVER_URLS
import lb_client

SCENARIOS = {
    "as_is": "Keep the current server states (set them in Server Lab)",
    "normal": "All servers healthy (recover-all)",
    "slow": "Target server slow (250 ms base latency)",
    "overload": "Target server overloaded (92% CPU, capacity 4, 15% errors)",
    "failure": "Target server hard-failed",
}


def apply_scenario(scenario: str, target: str) -> Dict:
    """Apply a scenario through the REAL worker APIs. Returns the list of
    servers that are degraded by construction (ground truth for routing
    accuracy)."""
    if scenario == "as_is":
        ws = lb_client.workers()
        degraded = [w["name"] for w in ws if w["status"] in ("FAILED", "OVERLOADED", "SLOW", "UNREACHABLE")]
        return {"applied": [], "degraded": degraded}
    applied = lb_client.recover_all()
    degraded = []
    if scenario in ("slow", "overload", "failure"):
        action = {"slow": "slow", "overload": "overload", "failure": "fail"}[scenario]
        applied.append(lb_client.worker_action(target, action))
        degraded = [target]
    time.sleep(1.0)  # let the CPU/memory burners take effect
    return {"applied": applied, "degraded": degraded}


class TrafficJob:
    def __init__(self, n: int, algorithm: str, scenario: str, target: str,
                 rate: float, concurrency: int, on_event=None):
        self.on_event = on_event
        self.id = uuid.uuid4().hex[:8]
        self.n, self.algorithm, self.scenario, self.target = n, algorithm, scenario, target
        self.rate, self.concurrency = rate, concurrency
        self.events: List[Dict] = []
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.state = "starting"
        self.sent = 0
        self.started_at = None
        self.finished_at = None
        self.degraded: List[str] = []
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _one(self, req_no: int):
        t_send = time.time()
        r = lb_client.route_request(self.algorithm)
        ev = {
            "id": req_no,
            "t_send": t_send - self.started_at,
            "t_done": time.time() - self.started_at,
            "latency_ms": r["latency_ms"],
            "success": r["success"],
            "status_code": r["status_code"],
            "selected_server": r.get("selected_server"),
            "q_values": r.get("q_values"),
            "algorithm": self.algorithm,
            "server_metrics": r.get("server_metrics"),
        }
        with self.lock:
            ev["seq"] = len(self.events) + 1
            self.events.append(ev)
        if self.on_event:
            try:
                self.on_event(r)
            except Exception:
                pass

    def _run(self):
        try:
            sc = apply_scenario(self.scenario, self.target)
            self.degraded = sc["degraded"]
            self.started_at = time.time()
            self.state = "running"
            interval = 1.0 / self.rate if self.rate > 0 else 0.0
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = []
                next_t = time.time()
                for i in range(1, self.n + 1):
                    if self.stop_flag.is_set():
                        break
                    # bounded in-flight requests
                    while sum(1 for f in futures if not f.done()) >= self.concurrency and not self.stop_flag.is_set():
                        time.sleep(0.005)
                    futures.append(pool.submit(self._one, i))
                    self.sent = i
                    if interval:
                        next_t += interval
                        time.sleep(max(0.0, next_t - time.time()))
                for f in futures:
                    try:
                        f.result()
                    except Exception:
                        pass
            self.state = "stopped" if self.stop_flag.is_set() else "completed"
        except Exception as e:
            self.error = str(e)
            self.state = "error"
        finally:
            self.finished_at = time.time()

    def stats(self) -> Dict:
        with self.lock:
            evs = list(self.events)
        if self.started_at is None:
            wall = 0.0
        else:
            wall = (self.finished_at or time.time()) - self.started_at
        s = lb_client.summarize(evs, wall, self.degraded or None) if evs else {"total_requests": 0}
        last_q = next((e["q_values"] for e in reversed(evs) if e.get("q_values")), None)
        return {
            "job_id": self.id, "state": self.state, "algorithm": self.algorithm,
            "algorithm_label": ALGORITHM_LABELS.get(self.algorithm, self.algorithm),
            "scenario": self.scenario, "target_server": self.target,
            "requested": self.n, "sent": self.sent, "completed": len(evs),
            "in_flight": max(0, self.sent - len(evs)),
            "rate": self.rate, "concurrency": self.concurrency,
            "elapsed_seconds": round(wall, 3), "error": self.error,
            "latest_q_values": last_q, "summary": s,
        }

    def events_since(self, seq: int, limit: int = 400) -> List[Dict]:
        with self.lock:
            out = self.events[seq: seq + limit]
        # strip bulky server_metrics from the event stream
        return [{k: v for k, v in e.items() if k != "server_metrics"} for e in out]


class TrafficManager:
    def __init__(self):
        self.job: Optional[TrafficJob] = None
        self.lock = threading.Lock()
        self.on_event = None

    def running(self) -> bool:
        return self.job is not None and self.job.state in ("starting", "running")

    def start(self, **kw) -> TrafficJob:
        with self.lock:
            if self.running():
                raise RuntimeError("A traffic job is already running - stop it first.")
            if kw["n"] < 1 or kw["n"] > MAX_TRAFFIC_REQUESTS:
                raise ValueError(f"requests must be between 1 and {MAX_TRAFFIC_REQUESTS}")
            if kw["algorithm"] not in BALANCER_URLS:
                raise ValueError("unknown algorithm")
            if kw["scenario"] not in SCENARIOS:
                raise ValueError("unknown scenario")
            if kw["target"] not in SERVER_URLS:
                raise ValueError("unknown target server")
            self.job = TrafficJob(**kw, on_event=self.on_event)
            self.job.start()
            return self.job

    def stop(self):
        if self.job:
            self.job.stop_flag.set()

    def reset(self):
        with self.lock:
            if self.running():
                raise RuntimeError("Stop the running job before resetting.")
            self.job = None


class ComparisonJob:
    def __init__(self, scenario, target, n, algorithms, rate, concurrency, on_done=None):
        self.id = uuid.uuid4().hex[:8]
        self.scenario, self.target, self.n = scenario, target, n
        self.algorithms, self.rate, self.concurrency = algorithms, rate, concurrency
        self.state = "running"
        self.current = None
        self.progress = {}
        self.results: Dict[str, Dict] = {}
        self.error = None
        self.on_done = on_done
        self.stop_flag = threading.Event()
        self.started = time.time()
        threading.Thread(target=self._run, daemon=True).start()

    def config(self):
        return {"scenario": self.scenario, "target_server": self.target, "requests": self.n,
                "algorithms": self.algorithms, "rate": self.rate, "concurrency": self.concurrency}

    def _run(self):
        try:
            for algo in self.algorithms:
                if self.stop_flag.is_set():
                    break
                self.current = algo
                lb_client.reset_servers()
                try:
                    lb_client._post(BALANCER_URLS[algo] + "/reset", 3, retries=0)
                except Exception:
                    pass
                job = TrafficJob(self.n, algo, self.scenario, self.target, self.rate, self.concurrency)
                job.start()
                while job.thread.is_alive():
                    if self.stop_flag.is_set():
                        job.stop_flag.set()
                    self.progress[algo] = len(job.events)
                    time.sleep(0.3)
                self.progress[algo] = len(job.events)
                st = job.stats()
                self.results[algo] = {**st["summary"], "algorithm": algo,
                                      "algorithm_label": ALGORITHM_LABELS[algo],
                                      "final_q_values": st["latest_q_values"]}
            lb_client.recover_all()
            self.state = "stopped" if self.stop_flag.is_set() else "completed"
            if self.on_done and self.results:
                self.on_done(self)
        except Exception as e:
            self.error, self.state = str(e), "error"
        finally:
            self.current = None

    def status(self):
        return {"job_id": self.id, "state": self.state, "current": self.current,
                "progress": self.progress, "config": self.config(), "results": self.results,
                "error": self.error, "elapsed_seconds": round(time.time() - self.started, 1)}
