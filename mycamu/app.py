#!/usr/bin/env python3
"""
mycamu/app.py
=============
TCE MyCamu - student portal + live load-balancing demonstration front end.

Student-facing (login required, session cookie):
  POST /api/auth/login            {student_id, password}
  POST /api/auth/logout
  GET  /api/auth/me
  GET  /api/dashboard
  GET  /api/courses               (+ registration/clash status for the student)
  GET  /api/courses/<code>
  POST /api/register/<code>       validate -> route through selected balancer -> save
  DELETE /api/register/<code>     drop a course
  GET  /api/timetable
  GET  /api/fees
  GET  /api/history

Load-balancing / technical (open, so they can be driven with curl):
  GET  /api/routing/dqn           DQN /status + live /predict + last decision
  GET  /api/routing/algorithm     current algorithm
  POST /api/routing/algorithm     {algorithm: dqn|round_robin|least_connections|aco}
  GET  /api/workers               real metrics + status of server1..server4
  POST /api/workers/<name>/<action>   action = slow|overload|fail|recover|normal
  POST /api/workers/recover-all
  POST /api/rush/<level>          level = normal|medium|high (synchronous burst)
  POST /api/traffic/start         {requests, algorithm, scenario, target_server, rate, concurrency}
  POST /api/traffic/stop
  POST /api/traffic/reset
  GET  /api/traffic/status?since=<seq>
  POST /api/compare/start         {scenario, requests, algorithms[], target_server, rate, concurrency}
  POST /api/compare/stop
  GET  /api/compare/status
  GET  /api/compare/history
  GET  /api/health                actually calls every service
  GET  /metrics                   Prometheus metrics of MyCamu itself
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

from flask import Flask, jsonify, request, send_from_directory, session
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from werkzeug.security import check_password_hash

import db
import lb_client
from config import (ALGORITHM_LABELS, BALANCER_URLS, DATA_DIR, DEFAULT_ALGORITHM, PORT,
                    SERVER_URLS)
from traffic import SCENARIOS, ComparisonJob, TrafficManager

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")


def _secret_key() -> str:
    env = os.environ.get("MYCAMU_SECRET_KEY")
    if env:
        return env
    os.makedirs(DATA_DIR, exist_ok=True)
    p = os.path.join(DATA_DIR, ".secret_key")
    if os.path.exists(p):
        return open(p).read().strip()
    k = secrets.token_hex(32)
    with open(p, "w") as f:
        f.write(k)
    return k


app.config.update(SECRET_KEY=_secret_key(), SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  JSON_SORT_KEYS=False)

db.init_db()
traffic = TrafficManager()
_compare = {"job": None}
_last_decision = {"value": None}
_reg_lock = threading.Lock()

# ---------------- Prometheus -----------------
M_REG = Counter("mycamu_registrations_total", "Course registration attempts", ["result"])
M_ROUTED = Counter("mycamu_routed_requests_total", "Requests MyCamu routed through a balancer",
                   ["algorithm", "server", "outcome"])
M_LAT = Histogram("mycamu_routed_request_latency_seconds", "End-to-end latency of routed requests", ["algorithm"])
M_LOGIN = Counter("mycamu_logins_total", "Login attempts", ["result"])


def _record(r):
    M_ROUTED.labels(algorithm=r["algorithm"], server=r.get("selected_server") or "none",
                    outcome="success" if r["success"] else "failure").inc()
    M_LAT.labels(algorithm=r["algorithm"]).observe(r["latency_ms"] / 1000.0)


def _traffic_event(r):
    _record(r)
    if r.get("selected_server"):
        _last_decision["value"] = {**r, "ts": time.time(), "source": "Traffic Control Center"}


traffic.on_event = _traffic_event


def current_algorithm() -> str:
    a = db.get_setting("algorithm", DEFAULT_ALGORITHM)
    return a if a in BALANCER_URLS else "dqn"


def err(msg, code=400, **extra):
    return jsonify({"success": False, "error": msg, **extra}), code


def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        sid = session.get("student_id")
        if not sid or not db.query_one("SELECT 1 FROM students WHERE student_id = ?", (sid,)):
            return err("Not logged in", 401)
        return fn(sid, *a, **kw)
    return wrapper


# ======================================================================
# Pages
# ======================================================================
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


# ======================================================================
# Auth
# ======================================================================
def public_student(s):
    return {k: s[k] for k in ("student_id", "name", "department", "semester", "max_credits", "email", "cgpa", "batch")}


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or request.form
    sid = (data.get("student_id") or "").strip().lower()
    pw = data.get("password") or ""
    s = db.query_one("SELECT * FROM students WHERE student_id = ?", (sid,))
    if not s or not check_password_hash(s["password_hash"], pw):
        M_LOGIN.labels(result="failure").inc()
        return err("Invalid student ID or password", 401)
    session.clear()
    session["student_id"] = sid
    M_LOGIN.labels(result="success").inc()
    return jsonify({"success": True, "student": public_student(s)})


@app.route("/api/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True})


@app.route("/api/auth/me")
@login_required
def me(sid):
    return jsonify({"success": True, "student": public_student(db.query_one("SELECT * FROM students WHERE student_id=?", (sid,)))})


@app.route("/api/auth/demo-accounts")
def demo_accounts():
    rows = db.query("SELECT student_id, name, department FROM students ORDER BY student_id LIMIT 6")
    return jsonify({"accounts": rows})


# ======================================================================
# Student data
# ======================================================================
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def _registered(sid):
    return db.query("SELECT c.*, r.registered_at, r.algorithm AS reg_algorithm, r.server AS reg_server, "
                    "r.latency_ms AS reg_latency_ms FROM registrations r JOIN courses c ON c.code = r.course_code "
                    "WHERE r.student_id = ? ORDER BY c.code", (sid,))


def _overlap(a, b):
    return a["day"] == b["day"] and a["start_time"] < b["end_time"] and b["start_time"] < a["end_time"]


def _clashes(sid, code):
    new_slots = db.course_slots(code)
    out = []
    for c in _registered(sid):
        if c["code"] == code:
            continue
        for s1 in db.course_slots(c["code"]):
            for s2 in new_slots:
                if _overlap(s1, s2):
                    out.append({"course": c["code"], "course_name": c["name"], "day": s1["day"],
                                "time": f"{s1['start_time']}-{s1['end_time']}"})
    return out


def _seats_taken(code):
    return db.query_one("SELECT COUNT(*) AS n FROM registrations WHERE course_code = ?", (code,))["n"]


def _course_view(c, sid):
    reg_codes = {r["code"] for r in _registered(sid)}
    taken = _seats_taken(c["code"])
    return {**c, "slots": db.course_slots(c["code"]), "seats_taken": taken,
            "seats_available": max(0, c["seats"] - taken), "registered": c["code"] in reg_codes,
            "clashes": [] if c["code"] in reg_codes else _clashes(sid, c["code"])}


@app.route("/api/dashboard")
@login_required
def dashboard(sid):
    s = db.query_one("SELECT * FROM students WHERE student_id=?", (sid,))
    regs = _registered(sid)
    credits = sum(c["credits"] for c in regs)
    upcoming = []
    for c in regs:
        for sl in db.course_slots(c["code"]):
            upcoming.append({**sl, "code": c["code"], "name": c["name"]})
    upcoming.sort(key=lambda x: (DAY_ORDER.index(x["day"]), x["start_time"]))
    today = time.strftime("%A")
    if today in DAY_ORDER:
        k = DAY_ORDER.index(today)
        upcoming.sort(key=lambda x: ((DAY_ORDER.index(x["day"]) - k) % 7, x["start_time"]))
    recent = db.query("SELECT * FROM registration_log WHERE student_id=? ORDER BY id DESC LIMIT 5", (sid,))
    return jsonify({
        "success": True, "student": public_student(s),
        "registered_credits": credits, "remaining_credits": s["max_credits"] - credits,
        "registered_courses": [{**c, "slots": db.course_slots(c["code"])} for c in regs],
        "upcoming_classes": upcoming[:6], "recent_activity": recent,
        "algorithm": current_algorithm(),
    })


@app.route("/api/courses")
@login_required
def courses(sid):
    return jsonify({"success": True, "courses": [_course_view(c, sid) for c in db.query("SELECT * FROM courses ORDER BY code")]})


@app.route("/api/courses/<code>")
@login_required
def course_detail(sid, code):
    c = db.query_one("SELECT * FROM courses WHERE code = ?", (code.upper(),))
    if not c:
        return err("Course not found", 404)
    return jsonify({"success": True, "course": _course_view(c, sid)})


@app.route("/api/register/<course_code>", methods=["POST"])
@login_required
def register(sid, course_code):
    """Validation -> REAL routed request -> persist.

    Response keeps the spec'd routing fields at top level
    (algorithm, latency_ms, q_values, selected_server, status_code) plus
    the list of checks, server states and the final result."""
    code = course_code.upper()
    algorithm = (request.get_json(silent=True) or {}).get("algorithm") or current_algorithm()
    if algorithm not in BALANCER_URLS:
        return err("unknown algorithm")
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        return ok

    with _reg_lock:  # serialise validation+insert so seats/credits can't race
        s = db.query_one("SELECT * FROM students WHERE student_id=?", (sid,))
        c = db.query_one("SELECT * FROM courses WHERE code=?", (code,))
        ok = check("student_verified", s is not None, f"{s['name']} ({sid})" if s else "unknown student")
        ok = ok and check("course_exists", c is not None, c["name"] if c else f"{code} does not exist")
        if ok:
            regs = _registered(sid)
            dup = any(r["code"] == code for r in regs)
            ok = check("duplicate_registration", not dup,
                       "already registered" if dup else "not registered yet")
        if ok:
            used = sum(r["credits"] for r in regs)
            ok = check("credit_limit", used + c["credits"] <= s["max_credits"],
                       f"{used} + {c['credits']} = {used + c['credits']} / max {s['max_credits']} credits")
        if ok:
            cl = _clashes(sid, code)
            ok = check("timetable_clash", not cl,
                       "no clash" if not cl else "clashes with " + ", ".join(f"{x['course']} ({x['day']} {x['time']})" for x in cl))
        if ok:
            left = c["seats"] - _seats_taken(code)
            ok = check("seat_availability", left > 0, f"{left} of {c['seats']} seats available")

        if not ok:
            failed = next(x for x in checks if not x["passed"])
            M_REG.labels(result="rejected").inc()
            db.log_attempt(sid, code, False, f"{failed['check']}: {failed['detail']}")
            return jsonify({"success": False, "result": "REGISTRATION FAILED", "stage": "validation",
                            "reason": failed["detail"], "failed_check": failed["check"], "checks": checks,
                            "algorithm": algorithm, "latency_ms": None, "q_values": None,
                            "selected_server": None, "status_code": None}), 409

        routing = lb_client.route_request(algorithm)
        _record(routing)
        _last_decision["value"] = {**routing, "ts": time.time(), "source": f"registration {sid} {code}"}
        check("request_routed", routing["reachable"],
              f"{ALGORITHM_LABELS[algorithm]} -> {routing.get('selected_server')}" if routing["reachable"]
              else routing.get("error", "balancer unreachable"))
        if routing["success"]:
            with db.tx() as conn:
                conn.execute("INSERT INTO registrations (student_id, course_code, algorithm, server, latency_ms) "
                             "VALUES (?,?,?,?,?)", (sid, code, algorithm, routing["selected_server"], routing["latency_ms"]))
            check("registration_saved", True, "stored in MyCamu database")
            M_REG.labels(result="success").inc()
            db.log_attempt(sid, code, True, "registered", routing)
            result, http = "REGISTRATION SUCCESSFUL", 200
            reason = f"Processed by {routing['selected_server']}"
        else:
            M_REG.labels(result="backend_failure").inc()
            reason = (f"Backend {routing.get('selected_server')} returned HTTP {routing['status_code']}"
                      if routing["reachable"] else "Load balancer unreachable")
            db.log_attempt(sid, code, False, reason, routing)
            result, http = "REGISTRATION FAILED", 502

    regs = _registered(sid)
    return jsonify({
        "success": routing["success"], "result": result, "reason": reason, "stage": "routing",
        "course": {"code": c["code"], "name": c["name"], "credits": c["credits"]},
        "checks": checks,
        # --- spec'd routing fields (unchanged names) ---
        "algorithm": algorithm, "latency_ms": routing["latency_ms"], "q_values": routing.get("q_values"),
        "selected_server": routing.get("selected_server"), "status_code": routing["status_code"],
        # --- additional real details ---
        "algorithm_label": ALGORITHM_LABELS[algorithm],
        "backend_response_time_ms": routing.get("backend_response_time_ms"),
        "decision_time_ms": routing.get("decision_time_ms"),
        "server_metrics_before": routing.get("server_metrics_before"),
        "server_metrics": routing.get("server_metrics"),
        "registered_credits": sum(r["credits"] for r in regs),
    }), http


@app.route("/api/register/<course_code>", methods=["DELETE"])
@login_required
def drop(sid, course_code):
    code = course_code.upper()
    with db.tx() as conn:
        n = conn.execute("DELETE FROM registrations WHERE student_id=? AND course_code=?", (sid, code)).rowcount
    if not n:
        return err("Not registered for this course", 404)
    db.log_attempt(sid, code, True, "dropped", action="drop")
    return jsonify({"success": True, "dropped": code})


@app.route("/api/timetable")
@login_required
def timetable(sid):
    entries = []
    for c in _registered(sid):
        for sl in db.course_slots(c["code"]):
            entries.append({**sl, "code": c["code"], "name": c["name"], "faculty": c["faculty"], "credits": c["credits"]})
    entries.sort(key=lambda x: (DAY_ORDER.index(x["day"]), x["start_time"]))
    return jsonify({"success": True, "days": DAY_ORDER, "entries": entries})


@app.route("/api/fees")
@login_required
def fees(sid):
    s = db.query_one("SELECT * FROM students WHERE student_id=?", (sid,))
    credits = sum(c["credits"] for c in _registered(sid))
    f = db.FEE_STRUCTURE
    items = [
        {"item": "Tuition fee", "amount": f["tuition"]},
        {"item": f"Course / lab fee ({credits} credits x Rs.{f['per_credit_lab_fee']})", "amount": credits * f["per_credit_lab_fee"]},
        {"item": "Examination fee", "amount": f["exam_fee"]},
        {"item": "Library fee", "amount": f["library_fee"]},
    ]
    total = sum(i["amount"] for i in items)
    payments = db.query("SELECT * FROM fee_payments WHERE student_id=? AND semester=? ORDER BY paid_on", (sid, s["semester"]))
    paid = sum(p["amount"] for p in payments)
    return jsonify({"success": True, "semester": s["semester"], "items": items, "total": total,
                    "payments": payments, "paid": paid, "balance": total - paid})


@app.route("/api/history")
@login_required
def history(sid):
    rows = db.query("SELECT r.*, c.name AS course_name FROM registration_log r LEFT JOIN courses c ON c.code = r.course_code "
                    "WHERE r.student_id=? ORDER BY r.id DESC LIMIT 200", (sid,))
    for r in rows:
        r["q_values"] = json.loads(r.pop("q_values_json")) if r.get("q_values_json") else None
    return jsonify({"success": True, "history": rows})


# ======================================================================
# Routing / DQN
# ======================================================================
@app.route("/api/routing/dqn")
def routing_dqn():
    st = lb_client.dqn_status()
    pred = lb_client.dqn_predict()
    return jsonify({
        "success": st.get("reachable", False),
        "current_algorithm": current_algorithm(),
        "status": st,
        "live_prediction": pred,
        "last_decision": _last_decision["value"],
    })


@app.route("/api/routing/algorithm", methods=["GET", "POST"])
def routing_algorithm():
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form
        a = (data.get("algorithm") or "").strip().lower().replace(" ", "_").replace("-", "_")
        a = {"least_connection": "least_connections", "rr": "round_robin", "lc": "least_connections"}.get(a, a)
        if a not in BALANCER_URLS:
            return err(f"algorithm must be one of {list(BALANCER_URLS)}")
        db.set_setting("algorithm", a)
    a = current_algorithm()
    return jsonify({"success": True, "algorithm": a, "label": ALGORITHM_LABELS[a],
                    "available": [{"id": k, "label": v} for k, v in ALGORITHM_LABELS.items()]})


# ======================================================================
# Workers (backend servers)
# ======================================================================
@app.route("/api/workers")
def api_workers():
    return jsonify({"success": True, "workers": lb_client.workers(), "timestamp": time.time()})


@app.route("/api/workers/recover-all", methods=["POST"])
def api_recover_all():
    res = lb_client.recover_all()
    time.sleep(0.3)
    return jsonify({"success": True, "results": res, "workers": lb_client.workers()})


@app.route("/api/workers/<name>/<action>", methods=["POST"])
def api_worker_action(name, action):
    n = lb_client.normalize_worker_name(name)
    if not n:
        return err(f"unknown worker '{name}'", 404)
    if action not in ("slow", "overload", "fail", "recover", "normal", "medium"):
        return err("action must be slow|overload|fail|recover|normal|medium")
    try:
        res = lb_client.worker_action(n, action)
    except Exception as e:
        return err(f"could not reach {n}: {e}", 502)
    time.sleep(0.5)  # let the change show up in the next /metrics sample
    return jsonify({"success": True, "result": res, "workers": lb_client.workers()})


# ======================================================================
# Rush (synchronous burst) - REAL requests through the selected balancer
# ======================================================================
RUSH_LEVELS = {"normal": (50, 5), "medium": (150, 15), "high": (300, 30)}


@app.route("/api/rush/<level>", methods=["POST"])
def rush(level):
    if level not in RUSH_LEVELS:
        return err(f"level must be one of {list(RUSH_LEVELS)}")
    if traffic.running():
        return err("A Traffic Control Center job is running - stop it first.", 409)
    data = request.get_json(silent=True) or {}
    n = int(data.get("requests", RUSH_LEVELS[level][0]))
    conc = int(data.get("concurrency", RUSH_LEVELS[level][1]))
    algorithm = data.get("algorithm") or current_algorithm()
    if algorithm not in BALANCER_URLS or not (1 <= n <= 2000) or not (1 <= conc <= 100):
        return err("invalid parameters")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as pool:
        results = list(pool.map(lambda _: lb_client.route_request(algorithm), range(n)))
    wall = time.time() - t0
    for r in results:
        _record(r)
    last = next((r for r in reversed(results) if r.get("selected_server")), None)
    if last:
        _last_decision["value"] = {**last, "ts": time.time(), "source": f"rush/{level}"}
    summary = lb_client.summarize(results, wall)
    return jsonify({"success": True, "level": level, "algorithm": algorithm,
                    "algorithm_label": ALGORITHM_LABELS[algorithm], "concurrency": conc,
                    **summary, "last_q_values": (last or {}).get("q_values")})


# ======================================================================
# Traffic Control Center
# ======================================================================
@app.route("/api/traffic/start", methods=["POST"])
def traffic_start():
    d = request.get_json(silent=True) or {}
    try:
        job = traffic.start(n=int(d.get("requests", 100)), algorithm=d.get("algorithm") or current_algorithm(),
                            scenario=d.get("scenario", "normal"), target=d.get("target_server", "server1"),
                            rate=float(d.get("rate", 20)), concurrency=int(d.get("concurrency", 10)))
    except (RuntimeError, ValueError) as e:
        return err(str(e), 409)
    return jsonify({"success": True, "job_id": job.id})


@app.route("/api/traffic/stop", methods=["POST"])
def traffic_stop():
    traffic.stop()
    return jsonify({"success": True})


@app.route("/api/traffic/reset", methods=["POST"])
def traffic_reset():
    try:
        traffic.reset()
    except RuntimeError as e:
        return err(str(e), 409)
    return jsonify({"success": True})


@app.route("/api/traffic/status")
def traffic_status():
    job = traffic.job
    if not job:
        return jsonify({"success": True, "state": "idle", "scenarios": SCENARIOS})
    since = int(request.args.get("since", 0))
    return jsonify({"success": True, **job.stats(), "events": job.events_since(since), "scenarios": SCENARIOS,
                    "degraded_servers": job.degraded})


# ======================================================================
# Algorithm comparison
# ======================================================================
def _save_comparison(job):
    with db.tx() as conn:
        conn.execute("INSERT INTO comparison_runs (config_json, results_json) VALUES (?, ?)",
                     (json.dumps(job.config()), json.dumps(job.results)))


@app.route("/api/compare/start", methods=["POST"])
def compare_start():
    j = _compare["job"]
    if (j and j.state == "running") or traffic.running():
        return err("Another traffic/comparison run is in progress.", 409)
    d = request.get_json(silent=True) or {}
    algos = [a for a in d.get("algorithms", list(BALANCER_URLS)) if a in BALANCER_URLS]
    n = int(d.get("requests", 100))
    sc = d.get("scenario", "normal")
    tgt = d.get("target_server", "server1")
    if not algos or sc not in SCENARIOS or tgt not in SERVER_URLS or not (1 <= n <= 1000):
        return err("invalid parameters")
    _compare["job"] = ComparisonJob(sc, tgt, n, algos, float(d.get("rate", 20)), int(d.get("concurrency", 10)),
                                    on_done=_save_comparison)
    return jsonify({"success": True, "job_id": _compare["job"].id})


@app.route("/api/compare/stop", methods=["POST"])
def compare_stop():
    if _compare["job"]:
        _compare["job"].stop_flag.set()
    return jsonify({"success": True})


@app.route("/api/compare/status")
def compare_status():
    j = _compare["job"]
    return jsonify({"success": True, **(j.status() if j else {"state": "idle"})})


@app.route("/api/compare/history")
def compare_history():
    rows = db.query("SELECT * FROM comparison_runs ORDER BY id DESC LIMIT 20")
    for r in rows:
        r["config"] = json.loads(r.pop("config_json"))
        r["results"] = json.loads(r.pop("results_json"))
    return jsonify({"success": True, "runs": rows})


# ======================================================================
# Health + metrics
# ======================================================================
@app.route("/api/health")
def api_health():
    details = lb_client.health_all()
    try:
        db.query_one("SELECT 1 AS ok")
        db_ok = True
    except Exception:
        db_ok = False
    services = {k: v["ok"] for k, v in details.items()}
    return jsonify({
        "success": True,
        "application": "TCE MyCamu",
        "services": services,
        "all_healthy": all(services.values()) and db_ok,
        "database": db_ok,
        "details": details,
        "timestamp": time.time(),
    })


@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    print(f"[mycamu] balancers: {BALANCER_URLS}")
    print(f"[mycamu] servers:   {SERVER_URLS}")
    print(f"[mycamu] database:  {db.DB_PATH}")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
