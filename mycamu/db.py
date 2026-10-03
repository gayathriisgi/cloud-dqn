"""
mycamu/db.py
============
SQLite persistence for MyCamu (students, courses, timetable slots,
registrations, registration history, fee payments, settings, saved
algorithm-comparison runs).

Schema initialisation is SAFE and IDEMPOTENT:
  * every table is created with CREATE TABLE IF NOT EXISTS
  * demo data is inserted only when the corresponding table is EMPTY
  * nothing is ever dropped or reset on start-up
so restarting the container (or the app) never loses registrations.

The database file lives at config.DB_PATH (default /app/data/mycamu.db in
Docker, mounted on the named volume `mycamu_data`).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager

from werkzeug.security import generate_password_hash

from config import DB_PATH

_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    student_id    TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    department    TEXT NOT NULL,
    semester      INTEGER NOT NULL,
    max_credits   INTEGER NOT NULL DEFAULT 24,
    email         TEXT,
    cgpa          REAL,
    batch         TEXT
);
CREATE TABLE IF NOT EXISTS courses (
    code        TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    credits     INTEGER NOT NULL,
    faculty     TEXT,
    department  TEXT,
    category    TEXT,
    seats       INTEGER NOT NULL DEFAULT 60,
    description TEXT
);
CREATE TABLE IF NOT EXISTS course_slots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    course_code TEXT NOT NULL REFERENCES courses(code),
    day         TEXT NOT NULL,
    start_time  TEXT NOT NULL,
    end_time    TEXT NOT NULL,
    room        TEXT
);
CREATE TABLE IF NOT EXISTS registrations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id    TEXT NOT NULL REFERENCES students(student_id),
    course_code   TEXT NOT NULL REFERENCES courses(code),
    registered_at TEXT NOT NULL DEFAULT (datetime('now')),
    algorithm     TEXT,
    server        TEXT,
    latency_ms    REAL,
    UNIQUE(student_id, course_code)
);
CREATE TABLE IF NOT EXISTS registration_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id      TEXT NOT NULL,
    course_code     TEXT NOT NULL,
    action          TEXT NOT NULL DEFAULT 'register',
    ts              TEXT NOT NULL DEFAULT (datetime('now')),
    success         INTEGER NOT NULL,
    reason          TEXT,
    algorithm       TEXT,
    selected_server TEXT,
    latency_ms      REAL,
    status_code     INTEGER,
    q_values_json   TEXT
);
CREATE TABLE IF NOT EXISTS fee_payments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  TEXT NOT NULL,
    semester    INTEGER NOT NULL,
    description TEXT NOT NULL,
    amount      REAL NOT NULL,
    paid_on     TEXT,
    reference   TEXT
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS comparison_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    config_json TEXT NOT NULL,
    results_json TEXT NOT NULL
);
"""

DEMO_PASSWORD = os.environ.get("MYCAMU_DEMO_PASSWORD", "tce@123")

DEMO_STUDENTS = [
    # id, name, dept, sem, max_credits, email, cgpa, batch
    ("student001", "Gayathri S", "Computer Science and Engineering", 7, 24, "student001@student.tce.edu", 8.9, "2023-2027"),
    ("student002", "Arjun R", "Computer Science and Engineering", 7, 24, "student002@student.tce.edu", 8.1, "2023-2027"),
    ("student003", "Priya M", "Information Technology", 7, 22, "student003@student.tce.edu", 9.2, "2023-2027"),
    ("student004", "Karthik V", "Electronics and Communication Engineering", 5, 22, "student004@student.tce.edu", 7.6, "2024-2028"),
    ("student005", "Divya K", "Computer Science and Engineering", 5, 24, "student005@student.tce.edu", 8.4, "2024-2028"),
]

DEMO_COURSES = [
    # code, name, credits, faculty, dept, category, seats, description, [(day, start, end, room)]
    ("CSE401", "Cloud Computing", 4, "Dr. R. Meenakshi", "CSE", "Core", 60,
     "Virtualisation, containers, orchestration, cloud storage and elastic load balancing.",
     [("Monday", "09:00", "10:00", "CS-301"), ("Wednesday", "11:00", "12:00", "CS-301"), ("Friday", "09:00", "10:00", "CS-301")]),
    ("CSE402", "Machine Learning", 4, "Dr. S. Vignesh", "CSE", "Core", 60,
     "Supervised and unsupervised learning, model evaluation, regularisation and neural networks.",
     [("Monday", "10:00", "11:00", "CS-204"), ("Thursday", "10:00", "11:00", "CS-204"), ("Friday", "14:00", "15:00", "CS-204")]),
    ("CSE403", "Distributed Systems", 3, "Dr. P. Anitha", "CSE", "Core", 60,
     "Consistency, consensus (Raft/Paxos), replication, fault tolerance and distributed transactions.",
     [("Tuesday", "09:00", "10:00", "CS-302"), ("Thursday", "11:00", "12:00", "CS-302")]),
    ("CSE404", "Deep Reinforcement Learning", 3, "Dr. K. Harish", "CSE", "Elective", 40,
     "MDPs, Q-learning, DQN, policy gradients and applications to systems control.",
     [("Tuesday", "14:00", "15:30", "AI-Lab"), ("Friday", "11:00", "12:00", "AI-Lab")]),
    ("CSE405", "Microservices Architecture", 3, "Dr. L. Saranya", "CSE", "Elective", 45,
     "Service decomposition, API gateways, messaging, observability and resilience patterns.",
     [("Wednesday", "09:00", "10:00", "CS-105"), ("Thursday", "14:00", "15:00", "CS-105")]),
    ("CSE406", "Database Engineering", 4, "Dr. M. Ramesh", "CSE", "Core", 60,
     "Query processing, indexing, transactions, concurrency control and distributed databases.",
     [("Tuesday", "10:00", "11:00", "CS-201"), ("Wednesday", "14:00", "15:00", "CS-201"), ("Saturday", "09:00", "10:00", "CS-201")]),
    ("CSE407", "Computer Networks Lab", 2, "Dr. J. Kavitha", "CSE", "Lab", 30,
     "Hands-on socket programming, packet analysis, routing and SDN experiments.",
     [("Monday", "14:00", "16:00", "Net-Lab")]),
    ("CSE408", "Information Security", 3, "Dr. T. Balaji", "CSE", "Elective", 50,
     "Cryptography, authentication, network security and secure software engineering.",
     [("Wednesday", "10:00", "11:00", "CS-303"), ("Friday", "10:00", "11:00", "CS-303")]),
    ("CSE409", "Big Data Analytics", 3, "Dr. N. Lakshmi", "CSE", "Elective", 50,
     "Hadoop, Spark, stream processing and large-scale data pipelines.",
     [("Monday", "10:00", "11:00", "CS-305"), ("Thursday", "09:00", "10:00", "CS-305")]),  # clashes with CSE402 (Mon 10:00)
    ("CSE410", "Edge and IoT Systems", 3, "Dr. G. Prakash", "CSE", "Elective", 40,
     "Edge computing, IoT protocols, fog orchestration and latency-aware offloading.",
     [("Tuesday", "11:00", "12:00", "IoT-Lab"), ("Saturday", "10:00", "11:00", "IoT-Lab")]),
    ("HSS401", "Professional Ethics", 2, "Dr. V. Sangeetha", "HSS", "Humanities", 120,
     "Engineering ethics, professional responsibility and case studies.",
     [("Saturday", "11:00", "12:00", "Seminar Hall")]),
    ("CSE499", "Capstone Project Phase I", 6, "Project Committee", "CSE", "Project", 60,
     "Team project: problem definition, literature survey, design and prototype.",
     [("Thursday", "15:00", "17:00", "Project Lab"), ("Friday", "15:00", "17:00", "Project Lab")]),
]

# Pre-existing registrations so dashboards/timetables are not empty.
DEMO_REGISTRATIONS = [
    ("student001", "CSE403"), ("student001", "CSE406"),
    ("student002", "CSE401"), ("student002", "CSE408"), ("student002", "CSE499"),
    ("student003", "CSE405"),
]

DEMO_PAYMENTS = [
    ("student001", 7, "Tuition fee - instalment 1", 45000, "2026-07-10", "TCE-FEE-70011"),
    ("student002", 7, "Tuition fee - full", 90000, "2026-07-05", "TCE-FEE-70012"),
    ("student003", 7, "Tuition fee - instalment 1", 45000, "2026-07-12", "TCE-FEE-70013"),
    ("student001", 7, "Examination fee", 2500, "2026-08-01", "TCE-EXM-70011"),
]

# Fee structure (per semester) - used to compute dues from REAL registered credits.
FEE_STRUCTURE = {"tuition": 90000, "per_credit_lab_fee": 750, "exam_fee": 2500, "library_fee": 1500}


def get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        _local.conn = conn
    return conn


@contextmanager
def tx():
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def query(sql, params=()):
    return [dict(r) for r in get_conn().execute(sql, params).fetchall()]


def query_one(sql, params=()):
    r = get_conn().execute(sql, params).fetchone()
    return dict(r) if r else None


def init_db() -> None:
    """Create missing tables and seed demo data ONLY into empty tables."""
    with tx() as c:
        c.executescript(SCHEMA)
        if c.execute("SELECT COUNT(*) FROM students").fetchone()[0] == 0:
            pw = generate_password_hash(DEMO_PASSWORD)
            c.executemany(
                "INSERT INTO students (student_id, name, password_hash, department, semester, max_credits, email, cgpa, batch)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                [(s[0], s[1], pw, *s[2:]) for s in DEMO_STUDENTS])
        if c.execute("SELECT COUNT(*) FROM courses").fetchone()[0] == 0:
            for code, name, cr, fac, dept, cat, seats, desc, slots in DEMO_COURSES:
                c.execute("INSERT INTO courses (code, name, credits, faculty, department, category, seats, description)"
                          " VALUES (?,?,?,?,?,?,?,?)", (code, name, cr, fac, dept, cat, seats, desc))
                c.executemany("INSERT INTO course_slots (course_code, day, start_time, end_time, room) VALUES (?,?,?,?,?)",
                              [(code, *sl) for sl in slots])
        if c.execute("SELECT COUNT(*) FROM registrations").fetchone()[0] == 0 and \
                c.execute("SELECT COUNT(*) FROM registration_log").fetchone()[0] == 0:
            c.executemany("INSERT OR IGNORE INTO registrations (student_id, course_code, algorithm, server)"
                          " VALUES (?,?, 'seed', NULL)", DEMO_REGISTRATIONS)
        if c.execute("SELECT COUNT(*) FROM fee_payments").fetchone()[0] == 0:
            c.executemany("INSERT INTO fee_payments (student_id, semester, description, amount, paid_on, reference)"
                          " VALUES (?,?,?,?,?,?)", DEMO_PAYMENTS)


def get_setting(key, default=None):
    r = query_one("SELECT value FROM settings WHERE key = ?", (key,))
    return r["value"] if r else default


def set_setting(key, value):
    with tx() as c:
        c.execute("INSERT INTO settings (key, value) VALUES (?, ?) "
                  "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))


def course_slots(code):
    return query("SELECT day, start_time, end_time, room FROM course_slots WHERE course_code = ? "
                 "ORDER BY CASE day WHEN 'Monday' THEN 1 WHEN 'Tuesday' THEN 2 WHEN 'Wednesday' THEN 3 "
                 "WHEN 'Thursday' THEN 4 WHEN 'Friday' THEN 5 ELSE 6 END, start_time", (code,))


def log_attempt(student_id, course_code, success, reason, routing=None, action="register"):
    routing = routing or {}
    with tx() as c:
        c.execute("INSERT INTO registration_log (student_id, course_code, action, success, reason, algorithm,"
                  " selected_server, latency_ms, status_code, q_values_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (student_id, course_code, action, 1 if success else 0, reason, routing.get("algorithm"),
                   routing.get("selected_server"), routing.get("latency_ms"), routing.get("status_code"),
                   json.dumps(routing.get("q_values")) if routing.get("q_values") else None))
