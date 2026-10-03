"""VALIDATION-ONLY shim for prometheus_client. Not shipped with the
project. Mirrors the subset of the real API used by this project
(Counter/Gauge/Histogram with .labels(...).inc()/.set()/.observe(),
start_http_server, generate_latest, CONTENT_TYPE_LATEST) so the real
project code can be imported and exercised end-to-end in a sandbox
that has no network access to install the real `prometheus_client`
package from PyPI.

Upgraded (2026-09-13) from a placeholder version that only tracked
values in memory but never actually served them: this version keeps
every created metric in a process-wide registry, renders REAL
Prometheus text-exposition format 0.0.4 (including real cumulative
histogram buckets, so histogram_quantile() works), and start_http_server
starts a REAL background HTTP server bound to the requested port that a
REAL Prometheus binary can scrape -- not a no-op. This lets the actual,
unmodified project code (load_balancer/*.py, servers/app.py) be scraped
by a genuine Prometheus server and visualized in Grafana, exactly as
the project's own docker-compose.yml / prometheus.yml already expect.
"""
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5,
                    0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float("inf"))

_REGISTRY = []
_REGISTRY_LOCK = threading.Lock()
_NAMES_SEEN = set()


def _sanitize(v) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class _Child:
    """Counter/Gauge child: a single scalar."""

    def __init__(self):
        self.value = 0.0
        self._lock = threading.Lock()

    def inc(self, amount=1.0):
        with self._lock:
            self.value += amount

    def dec(self, amount=1.0):
        with self._lock:
            self.value -= amount

    def set(self, value):
        with self._lock:
            self.value = float(value)

    def observe(self, value):  # Gauge/Counter used as .observe() defensively
        self.set(value)


class _HistChild:
    """Histogram child: real cumulative bucket counts + sum + count,
    exactly like the real prometheus_client's _HistogramMetricFamily."""

    def __init__(self, buckets):
        self.buckets = list(buckets)
        self.bucket_counts = [0] * len(self.buckets)
        self.sum = 0.0
        self.count = 0
        self._lock = threading.Lock()

    def observe(self, value):
        value = float(value)
        with self._lock:
            self.sum += value
            self.count += 1
            for i, upper in enumerate(self.buckets):
                if value <= upper:
                    self.bucket_counts[i] += 1
                    break  # per-bucket count only; render() computes the running cumulative total


class _Metric:
    prom_type = "gauge"

    def __init__(self, name, documentation, labelnames=(), buckets=DEFAULT_BUCKETS, **kwargs):
        self.name = name
        self.documentation = documentation
        self.labelnames = tuple(labelnames)
        self.buckets = tuple(buckets)
        self._children = {}
        self._default = self._make_child()
        with _REGISTRY_LOCK:
            # Real prometheus_client raises on duplicate registration;
            # a hot-reload/re-import here should just reuse the same
            # metric object rather than crash the app.
            if name in _NAMES_SEEN:
                for m in _REGISTRY:
                    if m.name == name:
                        self.__dict__ = m.__dict__
                        return
            _NAMES_SEEN.add(name)
            _REGISTRY.append(self)

    def _make_child(self):
        return _Child()

    def labels(self, *args, **kwargs):
        key = tuple(kwargs.get(l, args[i] if i < len(args) else "") for i, l in enumerate(self.labelnames))
        if key not in self._children:
            self._children[key] = self._make_child()
        return self._children[key]

    def inc(self, amount=1.0):
        self._default.inc(amount)

    def set(self, value):
        self._default.set(value)

    def observe(self, value):
        self._default.observe(value)

    # ------------------------------------------------------------
    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.documentation}", f"# TYPE {self.name} {self.prom_type}"]
        items = list(self._children.items())
        if not items:
            if self.labelnames:
                # A labeled metric with zero .labels(...) calls so far has
                # no series to report yet (matches real prometheus_client:
                # you cannot emit a value without its label values) --
                # emit HELP/TYPE only, no malformed empty-label line.
                return "\n".join(lines)
            items = [((), self._default)]
        for key, child in items:
            label_str = ""
            if self.labelnames:
                pairs = ",".join(f'{n}="{_sanitize(v)}"' for n, v in zip(self.labelnames, key))
                label_str = "{" + pairs + "}"
            if self.prom_type == "histogram":
                for upper, cum in zip(child.buckets, _cumulative(child.bucket_counts)):
                    le = "+Inf" if upper == float("inf") else _sanitize(upper)
                    if self.labelnames:
                        pairs = ",".join(f'{n}="{_sanitize(v)}"' for n, v in zip(self.labelnames, key))
                        bucket_labels = "{" + pairs + f',le="{le}"' + "}"
                    else:
                        bucket_labels = f'{{le="{le}"}}'
                    lines.append(f"{self.name}_bucket{bucket_labels} {cum}")
                lines.append(f"{self.name}_sum{label_str} {child.sum}")
                lines.append(f"{self.name}_count{label_str} {child.count}")
            else:
                lines.append(f"{self.name}{label_str} {child.value}")
        return "\n".join(lines)


def _cumulative(counts):
    total = 0
    out = []
    for c in counts:
        total += c
        out.append(total)
    return out


class Counter(_Metric):
    prom_type = "counter"


class Gauge(_Metric):
    prom_type = "gauge"


class Histogram(_Metric):
    prom_type = "histogram"

    def _make_child(self):
        return _HistChild(self.buckets)


class Summary(_Metric):
    prom_type = "summary"

    def _make_child(self):
        return _HistChild(())  # sum/count only, no buckets needed for this project

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.documentation}", f"# TYPE {self.name} summary"]
        items = list(self._children.items())
        if not items:
            if self.labelnames:
                return "\n".join(lines)
            items = [((), self._default)]
        for key, child in items:
            label_str = ""
            if self.labelnames:
                pairs = ",".join(f'{n}="{_sanitize(v)}"' for n, v in zip(self.labelnames, key))
                label_str = "{" + pairs + "}"
            lines.append(f"{self.name}_sum{label_str} {child.sum}")
            lines.append(f"{self.name}_count{label_str} {child.count}")
        return "\n".join(lines)


def generate_latest(registry=None) -> bytes:
    with _REGISTRY_LOCK:
        metrics = list(_REGISTRY)
    body = "\n".join(m.render() for m in metrics) + "\n"
    return body.encode("utf-8")


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 (matches BaseHTTPRequestHandler convention)
        body = generate_latest()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPE_LATEST)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # keep balancer/server stdout clean; this is scrape traffic


_SERVERS = {}


def start_http_server(port, addr="0.0.0.0"):
    """REAL background HTTP server (not a no-op) so a real Prometheus
    binary can actually scrape this process."""
    if port in _SERVERS:
        return _SERVERS[port]
    httpd = ThreadingHTTPServer((addr, port), _Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True, name=f"prom-metrics-{port}")
    t.start()
    _SERVERS[port] = httpd
    time.sleep(0.05)  # let the socket actually bind before the caller proceeds
    return httpd
