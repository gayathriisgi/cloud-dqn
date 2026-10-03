/* =====================================================================
   TCE MyCamu – single-page frontend (vanilla JS, no external libraries)
   ---------------------------------------------------------------------
   RULE: every number shown here comes from a backend response
   (MyCamu API -> balancers -> backend servers). The UI never invents
   routing decisions, Q-values, latencies or server metrics; animations
   only visualise events the backend already returned.
   ===================================================================== */
"use strict";

const S = { student: null, algorithm: "dqn", timers: [], pageToken: 0 };
const ALGO_LABEL = { dqn: "DQN", round_robin: "Round Robin", least_connections: "Least Connections", aco: "ACO" };
const ALGO_COLOR = { dqn: "#2563eb", round_robin: "#64748b", least_connections: "#d97706", aco: "#059669" };
const SERVERS = ["server1", "server2", "server3", "server4"];
const SRV_LABEL = (s) => s ? s.replace("server", "Server ") : "–";
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? "–" : Number(v).toFixed(d);
const pct = (v, d = 1) => (v === null || v === undefined) ? "–" : (v * 100).toFixed(d) + "%";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* ---------------- API ---------------- */
async function api(path, opts = {}) {
  const init = { method: opts.method || "GET", headers: {}, credentials: "same-origin" };
  if (opts.body !== undefined) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(opts.body); }
  let res;
  try { res = await fetch(path, init); }
  catch (e) { throw new Error("Network error – is MyCamu running?"); }
  let data = {};
  try { data = await res.json(); } catch (_) { /* non-JSON */ }
  if (res.status === 401 && !opts.allow401) { showLogin(); throw new Error("Please log in"); }
  data._status = res.status;
  if (!res.ok && !opts.allowError) throw Object.assign(new Error(data.error || data.reason || `HTTP ${res.status}`), { data });
  return data;
}

/* ---------------- toasts ---------------- */
function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = `toast ${kind}`; t.textContent = msg;
  $("#toasts").appendChild(t);
  setTimeout(() => t.remove(), 4200);
}

/* ---------------- timers (cleared on page change) ---------------- */
function every(ms, fn, immediate = true) {
  const token = S.pageToken;
  let stopped = false, handle = null;
  const tick = async () => {
    if (stopped || token !== S.pageToken || document.hidden) { if (!stopped && token === S.pageToken) handle = setTimeout(tick, ms); return; }
    try { await fn(); } catch (e) { /* keep polling */ }
    if (!stopped && token === S.pageToken) handle = setTimeout(tick, ms);
  };
  const stop = () => { stopped = true; clearTimeout(handle); };
  S.timers.push(stop);
  if (immediate) tick(); else handle = setTimeout(tick, ms);
  return stop;
}
function clearTimers() { S.timers.forEach((s) => s()); S.timers = []; }

/* ---------------- modal ---------------- */
function openModal(html, { onClose } = {}) {
  const root = $("#modal-root");
  root.innerHTML = `<div class="modal-back"><div class="modal">${html}</div></div>`;
  const close = () => { root.innerHTML = ""; onClose && onClose(); };
  $(".modal-back", root).addEventListener("click", (e) => { if (e.target.classList.contains("modal-back")) close(); });
  $$("[data-close]", root).forEach((b) => b.addEventListener("click", close));
  return { root: $(".modal", root), close };
}

/* ---------------- shared widgets ---------------- */
function badge(status) { return `<span class="badge ${esc(status)}"><span class="dot"></span>${esc(status)}</span>`; }

function bars(values, { lowerBetter = false, unit = "", digits = 1, colors = {} } = {}) {
  const entries = Object.entries(values).filter(([, v]) => v !== null && v !== undefined && !Number.isNaN(v));
  if (!entries.length) return `<div class="muted small">No data yet</div>`;
  const max = Math.max(...entries.map(([, v]) => Math.abs(v))) || 1;
  const best = entries.reduce((b, e) => (lowerBetter ? e[1] < b[1] : e[1] > b[1]) ? e : b)[0];
  return `<div class="bars">${entries.map(([k, v]) => `
    <div class="bar-row ${k === best && entries.length > 1 ? "best" : ""}">
      <span>${esc(ALGO_LABEL[k] || SRV_LABEL(k) || k)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.max(2, 100 * Math.abs(v) / max)}%;background:${colors[k] || ALGO_COLOR[k] || "#2563eb"}"></div></div>
      <span class="bar-val">${fmt(v, digits)}${unit}</span>
    </div>`).join("")}</div>`;
}

function qBars(q) {
  if (!q) return `<div class="muted small">No Q-values (the selected algorithm is not DQN, or no decision yet)</div>`;
  const vals = SERVERS.map((s) => q[s]).filter((v) => v !== undefined);
  const mn = Math.min(...vals), mx = Math.max(...vals);
  const span = (mx - mn) || 1;
  return `<div class="bars">${SERVERS.map((s) => {
    const v = q[s]; const best = v === mx;
    return `<div class="bar-row ${best ? "best" : ""}"><span class="${best ? "q-best" : ""}">${SRV_LABEL(s)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${12 + 88 * (v - mn) / span}%;background:${best ? "#2563eb" : "#93c5fd"}"></div></div>
      <span class="bar-val mono">${fmt(v, 4)}</span></div>`;
  }).join("")}</div><div class="muted small" style="margin-top:6px">Bar length is relative to the min/max Q-value of this decision; values are the raw network outputs.</div>`;
}

function serverCard(w, opts = {}) {
  const m = w.metrics || {};
  const st = w.status || "UNREACHABLE";
  const bar = (v, max = 100) => `<div class="mini-bar"><span style="width:${Math.min(100, (100 * (v || 0)) / max)}%"></span></div>`;
  return `<div class="card server-card ${st} ${opts.selected ? "selected" : ""}" data-server="${w.name}">
    <div class="row"><span class="sname">${SRV_LABEL(w.name)}</span><span class="spacer"></span>${badge(st)}</div>
    ${w.reachable === false ? `<div class="muted small" style="margin-top:8px">Unreachable: ${esc(w.error || "")}</div>` : `
    <div class="metric-list">
      <div><span>CPU</span><b>${fmt(m.cpu_usage)}%</b></div>
      <div><span>Memory</span><b>${fmt(m.memory_usage)}%</b></div>
      <div><span>Queue</span><b>${m.queue_length ?? "–"}</b></div>
      <div><span>Active</span><b>${m.active_requests ?? "–"}</b></div>
      <div><span>Latency</span><b>${fmt(m.average_response_time_ms)} ms</b></div>
      <div><span>Error rate</span><b>${pct(m.error_rate)}</b></div>
      <div><span>Req/s</span><b>${fmt(m.request_rate, 2)}</b></div>
      <div><span>Load</span><b>${fmt(m.load_pressure, 3)}</b></div>
    </div>
    ${bar(m.cpu_usage)}
    ${opts.extra || ""}
    ${opts.showWorkload && w.workload ? `<div class="muted small" style="margin-top:6px">workload: base ${w.workload.base_latency_ms} ms · capacity ${w.workload.capacity} · cpu ${w.workload.cpu_level}% · err ${w.workload.error_rate}</div>` : ""}`}
    ${opts.actions || ""}
  </div>`;
}

function sparkline(values, { color = "#2563eb", height = 70, maxV = null } = {}) {
  if (!values.length) return `<svg class="spark"></svg>`;
  const w = 300, h = height, mx = maxV || Math.max(...values, 1);
  const pts = values.map((v, i) => `${(i / Math.max(1, values.length - 1)) * w},${h - (v / mx) * (h - 6) - 3}`).join(" ");
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><polyline fill="none" stroke="${color}" stroke-width="2" points="${pts}"/></svg>`;
}

/* ---------------- auth / layout ---------------- */
function showLogin() {
  clearTimers(); S.student = null;
  $("#app-view").classList.add("hidden"); $("#login-view").classList.remove("hidden");
  api("/api/auth/demo-accounts", { allow401: true }).then((d) => {
    $("#demo-list").innerHTML = (d.accounts || []).map((a) => `<button type="button" data-id="${esc(a.student_id)}" title="${esc(a.name)} – ${esc(a.department)}">${esc(a.student_id)}</button>`).join("");
    $$("#demo-list button").forEach((b) => b.addEventListener("click", () => { $("#login-id").value = b.dataset.id; $("#login-pw").focus(); }));
  }).catch(() => {});
}

function showApp() {
  $("#login-view").classList.add("hidden"); $("#app-view").classList.remove("hidden");
  const s = S.student;
  $("#user-chip").innerHTML = `<div class="avatar">${esc(s.name[0])}</div><div class="nm">${esc(s.name)}<div class="muted small">${esc(s.student_id)}</div></div>`;
  api("/api/routing/algorithm").then((d) => { S.algorithm = d.algorithm; $("#algo-select").value = d.algorithm; }).catch(() => {});
  route();
}

$("#login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("#login-btn"); btn.disabled = true; btn.innerHTML = `<span class="spinner"></span> Signing in…`;
  $("#login-error").classList.add("hidden");
  try {
    const d = await api("/api/auth/login", { method: "POST", body: { student_id: $("#login-id").value, password: $("#login-pw").value }, allow401: true, allowError: true });
    if (!d.success) throw new Error(d.error || "Login failed");
    S.student = d.student; $("#login-pw").value = "";
    if (!location.hash || location.hash === "#/") location.hash = "#/dashboard";
    showApp(); toast(`Welcome, ${d.student.name}`, "ok");
  } catch (err) {
    $("#login-error").textContent = err.message; $("#login-error").classList.remove("hidden");
  } finally { btn.disabled = false; btn.textContent = "LOGIN"; }
});

$("#logout-btn").addEventListener("click", async () => {
  await api("/api/auth/logout", { method: "POST" }).catch(() => {});
  location.hash = "#/"; showLogin(); toast("Logged out");
});

$("#algo-select").addEventListener("change", async (e) => {
  try {
    const d = await api("/api/routing/algorithm", { method: "POST", body: { algorithm: e.target.value } });
    S.algorithm = d.algorithm; toast(`Routing algorithm set to ${d.label}`, "ok");
  } catch (err) { toast(err.message, "bad"); }
});
$("#menu-btn").addEventListener("click", () => $("#sidebar").classList.toggle("open"));

/* ---------------- router ---------------- */
const PAGES = {};
const TITLES = { dashboard: "Dashboard", courses: "Courses", course: "Course Details", enrollment: "Enrollment", timetable: "Timetable",
  fees: "Fees", history: "Registration History", dqn: "DQN Monitor", servers: "Server Monitor", lab: "Server Lab",
  traffic: "Traffic Control Center", compare: "Algorithm Comparison", health: "System Health" };

function route() {
  if (!S.student) return;
  clearTimers(); S.pageToken++;
  const parts = (location.hash.replace(/^#\/?/, "") || "dashboard").split("/");
  let page = parts[0];
  if (page === "courses" && parts[1]) page = "course";
  if (!PAGES[page]) page = "dashboard";
  $$(".sidebar nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === (page === "course" ? "courses" : page)));
  $("#page-title").textContent = TITLES[page];
  $("#sidebar").classList.remove("open");
  const el = $("#page");
  el.innerHTML = `<div class="grid g3"><div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div></div>`;
  const token = S.pageToken;
  Promise.resolve(PAGES[page](el, parts.slice(1), token)).catch((err) => {
    if (token === S.pageToken) el.innerHTML = `<div class="card empty">Could not load this page: ${esc(err.message)}</div>`;
  });
}
window.addEventListener("hashchange", route);

/* =====================================================================
   STUDENT PAGES
   ===================================================================== */
PAGES.dashboard = async (el) => {
  const d = await api("/api/dashboard");
  const s = d.student;
  const used = d.registered_credits, max = s.max_credits;
  el.innerHTML = `
  <div class="card hero" style="margin-bottom:16px">
    <div class="row"><div><div class="muted">Welcome back</div><h2 style="font-size:24px;margin-top:4px">${esc(s.name)}</h2>
      <div class="muted" style="margin-top:4px">${esc(s.student_id)} · ${esc(s.department)} · Semester ${s.semester} · Batch ${esc(s.batch || "")}</div></div>
      <span class="spacer"></span><a class="btn" href="#/courses">Browse courses →</a></div>
  </div>
  <div class="grid g4">
    <div class="card stat"><span class="label">Registered credits</span><span class="value">${used}</span><div class="progress"><span style="width:${(100 * used) / max}%"></span></div><span class="sub">of ${max} maximum</span></div>
    <div class="card stat"><span class="label">Remaining credits</span><span class="value">${d.remaining_credits}</span><span class="sub">available this semester</span></div>
    <div class="card stat"><span class="label">Registered courses</span><span class="value">${d.registered_courses.length}</span><span class="sub">${d.upcoming_classes.length ? "next: " + esc(d.upcoming_classes[0].code) : "no classes yet"}</span></div>
    <div class="card stat"><span class="label">CGPA</span><span class="value">${fmt(s.cgpa, 2)}</span><span class="sub">Routing via <b>${ALGO_LABEL[d.algorithm]}</b></span></div>
  </div>
  <div class="grid g2" style="margin-top:16px">
    <div class="card"><div class="card-head"><h3>Registered courses</h3><span class="spacer"></span><a href="#/enrollment" class="small">Manage →</a></div>
      ${d.registered_courses.length ? `<div class="table-wrap"><table><tr><th>Code</th><th>Course</th><th>Credits</th><th>Faculty</th></tr>
      ${d.registered_courses.map((c) => `<tr><td><b>${esc(c.code)}</b></td><td>${esc(c.name)}</td><td>${c.credits}</td><td>${esc(c.faculty)}</td></tr>`).join("")}</table></div>`
      : `<div class="empty">No courses yet. <a href="#/courses">Enroll now</a></div>`}
    </div>
    <div class="card"><div class="card-head"><h3>Upcoming timetable</h3><span class="spacer"></span><a href="#/timetable" class="small">Full timetable →</a></div>
      ${d.upcoming_classes.length ? `<table>${d.upcoming_classes.map((u) => `<tr><td><b>${esc(u.day)}</b></td><td>${esc(u.start_time)}–${esc(u.end_time)}</td><td>${esc(u.code)} · ${esc(u.name)}</td><td class="muted">${esc(u.room || "")}</td></tr>`).join("")}</table>`
      : `<div class="empty">Nothing scheduled.</div>`}
    </div>
  </div>
  <div class="card" style="margin-top:16px"><h3>Recent registration activity</h3>
    ${d.recent_activity.length ? `<table><tr><th>When</th><th>Course</th><th>Action</th><th>Result</th><th>Routed by</th><th>Server</th><th>Latency</th></tr>
    ${d.recent_activity.map((r) => `<tr><td class="muted">${esc(r.ts)}</td><td>${esc(r.course_code)}</td><td>${esc(r.action)}</td>
      <td>${r.success ? badge("HEALTHY").replace("HEALTHY", "OK") : `<span class="badge bad">FAILED</span>`} <span class="small muted">${esc(r.reason || "")}</span></td>
      <td>${esc(ALGO_LABEL[r.algorithm] || r.algorithm || "–")}</td><td>${esc(SRV_LABEL(r.selected_server))}</td><td>${r.latency_ms ? fmt(r.latency_ms) + " ms" : "–"}</td></tr>`).join("")}</table>`
    : `<div class="empty">No activity yet.</div>`}
  </div>`;
};

function slotText(slots) { return slots.map((s) => `${s.day.slice(0, 3)} ${s.start_time}–${s.end_time}`).join(", "); }

PAGES.courses = async (el) => {
  const d = await api("/api/courses");
  el.innerHTML = `
  <div class="toolbar">
    <input type="search" id="c-search" placeholder="Search by code, name or faculty…">
    <select id="c-cat"><option value="">All categories</option>${[...new Set(d.courses.map((c) => c.category))].map((c) => `<option>${esc(c)}</option>`).join("")}</select>
    <label class="row" style="flex-direction:row;gap:6px;font-weight:500"><input type="checkbox" id="c-avail"> Only available</label>
    <span class="spacer"></span><span class="muted small">Registrations are routed through <b id="c-algo">${ALGO_LABEL[S.algorithm]}</b> (change at top right)</span>
  </div>
  <div class="grid g-auto" id="c-grid"></div>`;
  const render = () => {
    const q = $("#c-search").value.toLowerCase(), cat = $("#c-cat").value, avail = $("#c-avail").checked;
    const list = d.courses.filter((c) => (!q || `${c.code} ${c.name} ${c.faculty}`.toLowerCase().includes(q)) && (!cat || c.category === cat) && (!avail || (!c.registered && !c.clashes.length && c.seats_available > 0)));
    $("#c-grid").innerHTML = list.map(courseCard).join("") || `<div class="card empty">No courses match.</div>`;
    bindCourseButtons($("#c-grid"));
  };
  ["input", "change"].forEach((ev) => { $("#c-search").addEventListener(ev, render); $("#c-cat").addEventListener(ev, render); $("#c-avail").addEventListener(ev, render); });
  render();
};

function courseStatus(c) {
  if (c.registered) return `<span class="badge ok">✓ Registered</span>`;
  if (c.seats_available <= 0) return `<span class="badge bad">Full</span>`;
  if (c.clashes.length) return `<span class="badge SLOW" title="Clashes with ${esc(c.clashes.map((x) => x.course).join(", "))}">⚠ Timetable clash</span>`;
  return `<span class="badge info">Available</span>`;
}

function courseCard(c) {
  return `<div class="card course-card">
    <div class="row"><span class="code">${esc(c.code)}</span><span class="spacer"></span>${courseStatus(c)}</div>
    <h4>${esc(c.name)}</h4>
    <div class="course-meta"><span>Credits: <b>${c.credits}</b> · ${esc(c.category)}</span><span>Faculty: ${esc(c.faculty)}</span>
      <span>Schedule: ${esc(slotText(c.slots))}</span><span>Seats: ${c.seats_available} / ${c.seats} available</span></div>
    <div class="actions"><a class="btn btn-outline btn-sm" href="#/courses/${esc(c.code)}">VIEW DETAILS</a>
      ${c.registered ? `<button class="btn btn-sm btn-ghost" disabled>Enrolled</button>` : `<button class="btn btn-primary btn-sm" data-enroll="${esc(c.code)}" data-name="${esc(c.name)}">ENROLL NOW</button>`}</div>
  </div>`;
}

function bindCourseButtons(root) {
  $$("[data-enroll]", root).forEach((b) => b.addEventListener("click", () => enrollFlow(b.dataset.enroll, b.dataset.name)));
}

PAGES.course = async (el, [code]) => {
  const { course: c } = await api(`/api/courses/${encodeURIComponent(code)}`);
  el.innerHTML = `<a href="#/courses" class="small">← All courses</a>
  <div class="grid g2" style="margin-top:12px">
    <div class="card"><div class="row"><span class="code" style="font-weight:800;color:var(--primary)">${esc(c.code)}</span><span class="spacer"></span>${courseStatus(c)}</div>
      <h2 style="margin:8px 0">${esc(c.name)}</h2><p class="muted">${esc(c.description || "")}</p>
      <table><tr><th>Credits</th><td>${c.credits}</td></tr><tr><th>Category</th><td>${esc(c.category)}</td></tr>
      <tr><th>Faculty</th><td>${esc(c.faculty)}</td></tr><tr><th>Department</th><td>${esc(c.department)}</td></tr>
      <tr><th>Seats</th><td>${c.seats_taken} taken · ${c.seats_available} available of ${c.seats}</td></tr></table>
      <div style="margin-top:14px">${c.registered ? `<button class="btn btn-danger" id="drop-btn">Drop course</button>` : `<button class="btn btn-primary" data-enroll="${esc(c.code)}" data-name="${esc(c.name)}">ENROLL NOW</button>`}</div>
    </div>
    <div class="card"><h3>Weekly schedule</h3><table><tr><th>Day</th><th>Time</th><th>Room</th></tr>
      ${c.slots.map((s) => `<tr><td>${esc(s.day)}</td><td>${esc(s.start_time)}–${esc(s.end_time)}</td><td>${esc(s.room || "")}</td></tr>`).join("")}</table>
      ${c.clashes.length ? `<div class="form-error" style="margin-top:12px">Timetable clash with: ${c.clashes.map((x) => `${esc(x.course)} (${esc(x.day)} ${esc(x.time)})`).join(", ")}</div>` : ""}
    </div>
  </div>`;
  bindCourseButtons(el);
  const drop = $("#drop-btn");
  if (drop) drop.addEventListener("click", async () => {
    if (!confirmInline(drop, "Click again to confirm drop")) return;
    try { await api(`/api/register/${c.code}`, { method: "DELETE" }); toast(`${c.code} dropped`, "ok"); route(); } catch (e) { toast(e.message, "bad"); }
  });
};

function confirmInline(btn, msg) {
  if (btn.dataset.confirm === "1") return true;
  btn.dataset.confirm = "1"; const old = btn.textContent; btn.textContent = msg;
  setTimeout(() => { btn.dataset.confirm = ""; btn.textContent = old; }, 3000);
  return false;
}

/* ---------- the visual registration flow ---------- */
async function enrollFlow(code, name) {
  const algo = S.algorithm;
  const STEPS = [
    ["student_verified", "Student verified"], ["course_exists", "Course found"], ["duplicate_registration", "Duplicate registration checked"],
    ["credit_limit", "Credit limit checked"], ["timetable_clash", "Timetable clash checked"], ["seat_availability", "Seat availability checked"],
    ["request_routed", `Request sent to load balancer (${ALGO_LABEL[algo]})`], ["registration_saved", "Registration saved"],
  ];
  const m = openModal(`
    <div class="modal-head"><h3>Registration Processing – ${esc(code)} ${esc(name || "")}</h3><span class="spacer"></span><button class="btn btn-ghost" data-close>✕</button></div>
    <div class="modal-body"><div class="grid g2">
      <div><div class="steps" id="r-steps">${STEPS.map(([k, t]) => `<div class="step" data-k="${k}"><div class="ic">•</div><div><div class="t">${esc(t)}</div><div class="d"></div></div></div>`).join("")}</div></div>
      <div><div class="card" style="box-shadow:none"><h3>🧠 Routing decision</h3><div id="r-routing" class="muted small">Waiting for the backend…</div></div></div>
    </div><div id="r-servers" style="margin-top:14px"></div><div id="r-result"></div></div>
    <div class="modal-foot"><a class="btn btn-outline" href="#/timetable" data-close>Open timetable</a><button class="btn btn-primary" data-close>Close</button></div>`, { onClose: () => route() });
  const stepEl = (k) => $(`.step[data-k="${k}"]`, m.root);
  stepEl("student_verified").classList.add("active"); stepEl("student_verified").querySelector(".ic").innerHTML = `<span class="spinner"></span>`;

  let d;
  try { d = await api(`/api/register/${encodeURIComponent(code)}`, { method: "POST", body: { algorithm: algo }, allowError: true }); }
  catch (e) { $("#r-result", m.root).innerHTML = `<div class="result-banner bad">REGISTRATION FAILED<br><small>${esc(e.message)}</small></div>`; return; }

  // Reveal the REAL check results one by one (the backend already ran them).
  for (const [k] of STEPS) {
    const c = (d.checks || []).find((x) => x.check === k);
    const el = stepEl(k);
    if (!c) { continue; }
    el.classList.add("active"); el.querySelector(".ic").innerHTML = `<span class="spinner"></span>`;
    await sleep(k === "request_routed" ? 650 : 280);
    el.classList.remove("active"); el.classList.add(c.passed ? "done" : "fail");
    el.querySelector(".ic").textContent = c.passed ? "✓" : "✕"; el.querySelector(".d").textContent = c.detail;
    if (k === "request_routed") renderRouting(d, m.root);
    if (!c.passed) break;
  }
  if (d.stage === "validation") $("#r-routing", m.root).innerHTML = `<div class="muted">Validation failed before routing – no request was sent to the load balancer.</div>`;
  const ok = d.success;
  $("#r-result", m.root).innerHTML = `<div class="result-banner ${ok ? "ok" : "bad"}">${esc(d.result || (ok ? "REGISTRATION SUCCESSFUL" : "REGISTRATION FAILED"))}<br><small style="font-weight:500">${esc(d.reason || "")}</small></div>`;
  toast(`${code}: ${d.result}`, ok ? "ok" : "bad");
}

function renderRouting(d, root) {
  const before = d.server_metrics_before || d.server_metrics || {};
  $("#r-routing", root).innerHTML = `
    <table><tr><th>Algorithm</th><td>${esc(d.algorithm_label || d.algorithm)}</td></tr>
    <tr><th>Selected server</th><td><b style="color:var(--primary)">${esc(SRV_LABEL(d.selected_server))}</b></td></tr>
    <tr><th>End-to-end latency</th><td>${fmt(d.latency_ms, 1)} ms</td></tr>
    <tr><th>Backend processing</th><td>${fmt(d.backend_response_time_ms, 1)} ms</td></tr>
    <tr><th>Decision time</th><td>${d.decision_time_ms != null ? fmt(d.decision_time_ms, 2) + " ms" : "–"}</td></tr>
    <tr><th>HTTP status</th><td>${d.status_code}</td></tr></table>
    <div class="section-title" style="margin-top:12px">DQN Q-values</div>${qBars(d.q_values)}`;
  $("#r-servers", root).innerHTML = `<div class="section-title">Server states the algorithm analysed (collected just before routing)</div>
    <div class="grid g4">${SERVERS.map((s) => serverCard({ name: s, metrics: before[s], reachable: !!before[s], status: statusOf(before[s]) }, { selected: s === d.selected_server })).join("")}</div>`;
}

function statusOf(m) {  // same thresholds as the backend's lb_client.classify (display only)
  if (!m) return "UNREACHABLE";
  if (m.manually_failed || !m.health) return "FAILED";
  const cap = Math.max(1, m.capacity || 12), lat = m.average_response_time_ms || 0;
  if (m.cpu_usage >= 80 || m.queue_length >= cap || m.error_rate >= 0.1) return "OVERLOADED";
  if (lat >= 200) return "SLOW";
  if (m.load_pressure >= 0.4 || lat >= 100) return "BUSY";
  return "HEALTHY";
}

PAGES.enrollment = async (el) => {
  const [dash, cs] = await Promise.all([api("/api/dashboard"), api("/api/courses")]);
  const s = dash.student;
  const avail = cs.courses.filter((c) => !c.registered);
  el.innerHTML = `
  <div class="grid g3">
    <div class="card stat"><span class="label">Credits</span><span class="value">${dash.registered_credits} / ${s.max_credits}</span><div class="progress"><span style="width:${(100 * dash.registered_credits) / s.max_credits}%"></span></div></div>
    <div class="card stat"><span class="label">Registration routing</span><span class="value" style="font-size:20px">${ALGO_LABEL[S.algorithm]}</span><span class="sub">every enrollment is a real routed request</span></div>
    <div class="card stat"><span class="label">Available courses</span><span class="value">${avail.filter((c) => !c.clashes.length && c.seats_available > 0).length}</span><span class="sub">without clashes</span></div>
  </div>
  <div class="card" style="margin-top:16px"><h3>My enrolled courses</h3>
    ${dash.registered_courses.length ? `<table><tr><th>Code</th><th>Course</th><th>Credits</th><th>Schedule</th><th></th></tr>
    ${dash.registered_courses.map((c) => `<tr><td><b>${esc(c.code)}</b></td><td>${esc(c.name)}</td><td>${c.credits}</td><td class="small">${esc(slotText(c.slots))}</td>
      <td><button class="btn btn-outline btn-sm" data-drop="${esc(c.code)}">Drop</button></td></tr>`).join("")}</table>` : `<div class="empty">Not enrolled in any course yet.</div>`}
  </div>
  <div class="section-title">Enroll in a course</div>
  <div class="grid g-auto">${avail.map(courseCard).join("")}</div>`;
  bindCourseButtons(el);
  $$("[data-drop]", el).forEach((b) => b.addEventListener("click", async () => {
    if (!confirmInline(b, "Confirm?")) return;
    try { await api(`/api/register/${b.dataset.drop}`, { method: "DELETE" }); toast(`${b.dataset.drop} dropped`, "ok"); route(); } catch (e) { toast(e.message, "bad"); }
  }));
};

PAGES.timetable = async (el) => {
  const d = await api("/api/timetable");
  const hours = ["09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00"];
  const cell = (day, h) => d.entries.filter((e) => e.day === day && e.start_time.slice(0, 2) === h.slice(0, 2));
  el.innerHTML = `<div class="card" style="margin-bottom:14px"><div class="row"><b>${d.entries.length}</b> weekly sessions ·
    ${[...new Set(d.entries.map((e) => e.code))].length} courses<span class="spacer"></span><a class="btn btn-outline btn-sm" href="#/courses">+ Add course</a></div></div>
  <div class="tt"><div class="hd"></div>${d.days.map((x) => `<div class="hd">${x}</div>`).join("")}
  ${hours.map((h) => `<div class="hr">${h}</div>${d.days.map((day) => `<div>${cell(day, h).map((e) => `<div class="cls" title="${esc(e.faculty)}"><b>${esc(e.code)}</b>${esc(e.name)}<div class="muted">${esc(e.start_time)}–${esc(e.end_time)} · ${esc(e.room || "")}</div></div>`).join("")}</div>`).join("")}`).join("")}
  </div>`;
};

PAGES.fees = async (el) => {
  const d = await api("/api/fees");
  const inr = (v) => "₹" + Number(v).toLocaleString("en-IN");
  el.innerHTML = `<div class="grid g3">
    <div class="card stat"><span class="label">Total due (Sem ${d.semester})</span><span class="value">${inr(d.total)}</span></div>
    <div class="card stat"><span class="label">Paid</span><span class="value" style="color:var(--success)">${inr(d.paid)}</span></div>
    <div class="card stat"><span class="label">Balance</span><span class="value" style="color:${d.balance > 0 ? "var(--danger)" : "var(--success)"}">${inr(d.balance)}</span></div></div>
  <div class="grid g2" style="margin-top:16px">
    <div class="card"><h3>Fee breakdown</h3><table>${d.items.map((i) => `<tr><td>${esc(i.item)}</td><td style="text-align:right">${inr(i.amount)}</td></tr>`).join("")}
      <tr><th>Total</th><th style="text-align:right">${inr(d.total)}</th></tr></table><div class="muted small" style="margin-top:8px">Course/lab fee is computed from your currently registered credits.</div></div>
    <div class="card"><h3>Payments</h3>${d.payments.length ? `<table><tr><th>Date</th><th>Description</th><th>Reference</th><th>Amount</th></tr>
      ${d.payments.map((p) => `<tr><td>${esc(p.paid_on)}</td><td>${esc(p.description)}</td><td class="mono small">${esc(p.reference)}</td><td>${inr(p.amount)}</td></tr>`).join("")}</table>` : `<div class="empty">No payments recorded.</div>`}</div>
  </div>`;
};

PAGES.history = async (el) => {
  const d = await api("/api/history");
  el.innerHTML = `<div class="card"><div class="table-wrap"><table><tr><th>Time</th><th>Course</th><th>Action</th><th>Result</th><th>Reason</th><th>Algorithm</th><th>Server</th><th>Latency</th><th>Q-values</th></tr>
  ${d.history.map((r) => `<tr><td class="muted small">${esc(r.ts)}</td><td><b>${esc(r.course_code)}</b><div class="small muted">${esc(r.course_name || "")}</div></td><td>${esc(r.action)}</td>
    <td>${r.success ? `<span class="badge ok">SUCCESS</span>` : `<span class="badge bad">FAILED</span>`}</td><td class="small">${esc(r.reason || "")}</td>
    <td>${esc(ALGO_LABEL[r.algorithm] || "–")}</td><td>${esc(SRV_LABEL(r.selected_server))}</td><td>${r.latency_ms ? fmt(r.latency_ms) + " ms" : "–"}</td>
    <td class="mono small">${r.q_values ? SERVERS.map((s) => `${s.slice(-1)}:${fmt(r.q_values[s], 3)}`).join(" ") : "–"}</td></tr>`).join("") || `<tr><td colspan="9" class="empty">No registration attempts yet.</td></tr>`}
  </table></div></div>`;
};
