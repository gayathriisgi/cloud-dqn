/* =====================================================================
   TECHNICAL PAGES: DQN Monitor, Server Monitor, Server Lab,
   Traffic Control Center, Algorithm Comparison, System Health
   All data comes from /api/* (real balancer + server responses).
   ===================================================================== */
"use strict";

/* ---------------- DQN MONITOR ---------------- */
PAGES.dqn = async (el) => {
  el.innerHTML = `
  <div class="grid g4" id="dqn-status"></div>
  <div class="grid g2" style="margin-top:16px">
    <div class="card"><div class="card-head"><h3>🧠 Live Q-values</h3><span class="spacer"></span><span class="badge info" id="dqn-live">live</span></div>
      <div id="dqn-q"></div><div id="dqn-would" style="margin-top:12px"></div>
      <div class="muted small" style="margin-top:8px">From <span class="mono">GET /predict</span> on the DQN balancer: the real network evaluated on the current real server metrics (no request is routed).</div></div>
    <div class="card"><h3>Last routed decision</h3><div id="dqn-last" class="muted">No routed request yet – enroll in a course or run traffic.</div></div>
  </div>
  <div class="section-title">Server states seen by the DQN</div>
  <div class="grid g4" id="dqn-servers"></div>
  <div class="card" style="margin-top:16px"><h3>State vector fed to the network (46 normalised features)</h3><div id="dqn-heat"></div></div>`;
  const FEAT = ["cpu", "mem", "queue", "active", "lat", "rate", "thr", "err", "health", "load"];
  every(2000, async () => {
    const d = await api("/api/routing/dqn");
    const st = d.status || {};
    $("#dqn-status").innerHTML = `
      <div class="card stat"><span class="label">Algorithm</span><span class="value">${esc(st.algorithm || "DQN")}</span><span class="sub">${st.reachable ? badge("HEALTHY") : badge("UNREACHABLE")}</span></div>
      <div class="card stat"><span class="label">Model</span><span class="value" style="color:${st.model_loaded ? "var(--success)" : "var(--danger)"}">${st.model_loaded ? "Loaded" : "Not loaded"}</span><span class="sub" title="${esc(st.load_error || "")}">${esc((st.model_path || "").split("/").pop())}</span></div>
      <div class="card stat"><span class="label">State / action dim</span><span class="value">${st.state_dim ?? "–"} / ${st.action_dim ?? "–"}</span><span class="sub">steps trained: ${st.steps_done ?? "–"}</span></div>
      <div class="card stat"><span class="label">Epsilon</span><span class="value">${fmt(st.epsilon, 2)}</span><span class="sub">training ε at save: ${fmt(st.training_epsilon_at_save, 3)}</span></div>`;
    const p = d.live_prediction || {};
    if (p.available) {
      $("#dqn-q").innerHTML = qBars(p.q_values);
      $("#dqn-would").innerHTML = `Policy would select: <b style="color:var(--primary)">${esc(SRV_LABEL(p.would_select))}</b> <span class="muted small">(inference ${fmt(p.decision_time_ms, 2)} ms)</span>`;
      $("#dqn-servers").innerHTML = SERVERS.map((s) => serverCard({ name: s, metrics: p.server_metrics[s], reachable: true, status: statusOf(p.server_metrics[s]) }, { selected: s === p.would_select })).join("");
      const sv = p.state_vector || [];
      const color = (v) => `hsl(${220 - 220 * v}, 85%, ${92 - 40 * v}%)`;
      $("#dqn-heat").innerHTML = SERVERS.map((s, i) => `<div class="heat-labels"><b>${SRV_LABEL(s)}</b><div class="heat">${FEAT.map((f, j) => {
        const v = sv[i * 10 + j] ?? 0; return `<div style="background:${color(f === "health" ? 1 - v : v)}" title="${f}=${v}${f === "health" ? " (1 = healthy)" : ""}">${f}<br>${v.toFixed(2)}</div>`; }).join("")}</div></div>`).join("")
        + `<div class="heat-labels"><b>Global</b><div class="heat">${["tot_rate", "tot_active", "lat", "thr", "err", "intensity"].map((f, j) => {
          const v = sv[40 + j] ?? 0; return `<div style="background:${color(v)}">${f}<br>${v.toFixed(2)}</div>`; }).join("")}</div></div>`;
    } else {
      $("#dqn-q").innerHTML = `<div class="muted">${esc(p.error || "Live prediction unavailable")}</div>`;
    }
    const L = d.last_decision;
    if (L) {
      $("#dqn-last").innerHTML = `<table><tr><th>Source</th><td>${esc(L.source)}</td></tr><tr><th>Algorithm</th><td>${esc(ALGO_LABEL[L.algorithm] || L.algorithm)}</td></tr>
        <tr><th>Selected</th><td><b>${esc(SRV_LABEL(L.selected_server))}</b></td></tr><tr><th>Latency</th><td>${fmt(L.latency_ms)} ms</td></tr>
        <tr><th>Status</th><td>${L.status_code}</td></tr><tr><th>When</th><td>${new Date(L.ts * 1000).toLocaleTimeString()}</td></tr></table>
        <div class="section-title">Q-values of that decision</div>${qBars(L.q_values)}`;
    }
  });
};

/* ---------------- SERVER MONITOR ---------------- */
PAGES.servers = async (el) => {
  const hist = Object.fromEntries(SERVERS.map((s) => [s, { lat: [], cpu: [] }]));
  el.innerHTML = `<div class="row" style="margin-bottom:12px"><span class="muted">Live metrics from each backend's <span class="mono">/metrics</span>, refreshed every 2 s.</span><span class="spacer"></span><a class="btn btn-outline btn-sm" href="#/lab">Open Server Lab →</a></div>
  <div class="grid g4" id="sm-cards"></div>
  <div class="grid g2" style="margin-top:16px"><div class="card"><h3>Average latency (ms) – last 2 minutes</h3><div id="sm-lat"></div></div>
  <div class="card"><h3>CPU usage (%) – last 2 minutes</h3><div id="sm-cpu"></div></div></div>`;
  const colors = { server1: "#2563eb", server2: "#d97706", server3: "#059669", server4: "#db2777" };
  const multi = (key, maxV) => {
    const all = SERVERS.flatMap((s) => hist[s][key]); const mx = maxV || Math.max(10, ...all);
    return `<svg class="spark" style="height:150px" viewBox="0 0 300 150" preserveAspectRatio="none">${SERVERS.map((s) => {
      const v = hist[s][key]; if (!v.length) return "";
      return `<polyline fill="none" stroke="${colors[s]}" stroke-width="2" points="${v.map((x, i) => `${(i / 59) * 300},${150 - (x / mx) * 144 - 3}`).join(" ")}"/>`; }).join("")}</svg>
      <div class="row small">${SERVERS.map((s) => `<span><span class="dot" style="color:${colors[s]}"></span> ${SRV_LABEL(s)}</span>`).join("")}<span class="spacer"></span><span class="muted">max ${fmt(mx, 0)}</span></div>`;
  };
  every(2000, async () => {
    const d = await api("/api/workers");
    d.workers.forEach((w) => { const m = w.metrics || {}; const h = hist[w.name];
      h.lat.push(m.average_response_time_ms || 0); h.cpu.push(m.cpu_usage || 0);
      if (h.lat.length > 60) { h.lat.shift(); h.cpu.shift(); } });
    $("#sm-cards").innerHTML = d.workers.map((w) => serverCard(w, { showWorkload: true })).join("");
    $("#sm-lat").innerHTML = multi("lat"); $("#sm-cpu").innerHTML = multi("cpu", 100);
  });
};

/* ---------------- SERVER LAB ---------------- */
PAGES.lab = async (el) => {
  el.innerHTML = `
  <div class="card" style="margin-bottom:16px"><div class="row"><div><h3 style="margin:0">Apply real conditions to the backend servers</h3>
    <div class="muted small" style="margin-top:4px">Buttons call the servers' own control APIs (POST /load, /fail, /recover) through MyCamu. Status is re-read from /metrics after every action.</div></div>
    <span class="spacer"></span><button class="btn btn-success" id="lab-recover">⟲ RECOVER ALL</button></div></div>
  <div class="grid g4" id="lab-cards"></div>
  <div class="card" style="margin-top:16px"><div class="card-head"><h3>Guided failure demonstration</h3><span class="spacer"></span>
    <label style="flex-direction:row;align-items:center">Algorithm <select id="demo-algo">${Object.entries(ALGO_LABEL).map(([k, v]) => `<option value="${k}" ${k === S.algorithm ? "selected" : ""}>${v}</option>`).join("")}</select></label>
    <label style="flex-direction:row;align-items:center">Requests <input id="demo-n" type="number" value="100" min="10" max="500" style="width:80px"></label></div>
    <div class="demo-steps" id="demo-steps"></div>
    <div class="table-wrap" style="margin-top:14px"><table id="demo-table"><tr><th>Step</th><th>Success</th><th>Avg latency</th><th>P95</th><th>Throughput</th><th>Distribution (s1 / s2 / s3 / s4)</th><th>Final Q-values</th></tr></table></div>
  </div>`;
  const ACTIONS = [["normal", "NORMAL", "btn-outline"], ["slow", "SLOW", "btn-orange"], ["overload", "OVERLOAD", "btn-warn"], ["fail", "FAILURE", "btn-danger"], ["recover", "RECOVER", "btn-success"]];
  const render = (workers) => {
    $("#lab-cards").innerHTML = workers.map((w) => serverCard(w, { showWorkload: true, actions: `<div class="lab-actions">${ACTIONS.map(([a, t, c]) => `<button class="btn btn-sm ${c}" data-act="${a}" data-srv="${w.name}">${t}</button>`).join("")}</div>` })).join("");
    $$("[data-act]", el).forEach((b) => b.addEventListener("click", async () => {
      b.disabled = true; b.innerHTML = `<span class="spinner"></span>`;
      try { const r = await api(`/api/workers/${b.dataset.srv}/${b.dataset.act}`, { method: "POST" }); render(r.workers);
        toast(`${SRV_LABEL(b.dataset.srv)} → ${b.dataset.act.toUpperCase()} (now ${r.workers.find((w) => w.name === b.dataset.srv).status})`, "ok");
      } catch (e) { toast(e.message, "bad"); b.disabled = false; }
    }));
  };
  $("#lab-recover").addEventListener("click", async () => {
    try { const r = await api("/api/workers/recover-all", { method: "POST" }); render(r.workers); toast("All servers recovered", "ok"); } catch (e) { toast(e.message, "bad"); }
  });
  every(3000, async () => { const d = await api("/api/workers"); if (!$("#lab-cards .spinner")) render(d.workers); });

  // Guided demo – each step issues REAL actions / REAL traffic jobs
  const STEPS = [
    ["Normal traffic", "Recover all servers and send N requests", { scenario: "normal" }],
    ["Make Server 1 slow", "POST /api/workers/server1/slow", { action: ["server1", "slow"] }],
    ["Traffic with slow Server 1", "Send N requests with server1 still slow", { scenario: "as_is" }],
    ["Recover", "POST /api/workers/recover-all", { recover: true }],
    ["Traffic after recovery", "Send N requests again", { scenario: "as_is" }],
  ];
  $("#demo-steps").innerHTML = STEPS.map(([t, dsc], i) => `<div class="demo-step" data-i="${i}"><div class="n">${i + 1}</div><div><b>${t}</b><div class="muted small">${dsc}</div></div><button class="btn btn-primary btn-sm" data-step="${i}">Run</button></div>`).join("");
  $$("[data-step]", el).forEach((b) => b.addEventListener("click", async () => {
    const i = +b.dataset.step, [title, , cfg] = STEPS[i];
    $$("[data-step]", el).forEach((x) => (x.disabled = true)); b.innerHTML = `<span class="spinner"></span>`;
    try {
      if (cfg.action) { const r = await api(`/api/workers/${cfg.action[0]}/${cfg.action[1]}`, { method: "POST" }); render(r.workers); }
      else if (cfg.recover) { const r = await api("/api/workers/recover-all", { method: "POST" }); render(r.workers); }
      else {
        const algo = $("#demo-algo").value, n = +$("#demo-n").value || 100;
        await api("/api/traffic/reset", { method: "POST", allowError: true });
        await api("/api/traffic/start", { method: "POST", body: { requests: n, algorithm: algo, scenario: cfg.scenario, rate: 20, concurrency: 10 } });
        let st; const tok = S.pageToken;
        do { await sleep(700); st = await api("/api/traffic/status?since=999999"); b.innerHTML = `${st.completed}/${st.requested}`; }
        while (["starting", "running"].includes(st.state) && tok === S.pageToken);
        const s = st.summary || {};
        $("#demo-table").insertAdjacentHTML("beforeend", `<tr><td><b>${i + 1}. ${esc(title)}</b><div class="small muted">${esc(ALGO_LABEL[algo])}</div></td><td>${pct(s.success_rate)}</td><td>${fmt(s.avg_latency_ms)} ms</td>
          <td>${fmt(s.p95_latency_ms)} ms</td><td>${fmt(s.throughput_rps, 2)} rps</td><td>${SERVERS.map((x) => s.server_distribution ? s.server_distribution[x] : "–").join(" / ")}</td>
          <td class="mono small">${st.latest_q_values ? SERVERS.map((x) => fmt(st.latest_q_values[x], 3)).join(" / ") : "–"}</td></tr>`);
      }
      $(`.demo-step[data-i="${i}"]`, el).classList.add("done"); toast(`Step ${i + 1} done`, "ok");
    } catch (e) { toast(e.message, "bad"); }
    $$("[data-step]", el).forEach((x) => { x.disabled = false; x.textContent = "Run"; });
  }));
};

/* ---------------- TRAFFIC CONTROL CENTER ---------------- */
PAGES.traffic = async (el) => {
  const cfg = { n: 100, algorithm: S.algorithm, scenario: "normal", target: "server1", rate: 20, concurrency: 10 };
  el.innerHTML = `
  <div class="card" style="margin-bottom:16px"><div class="grid g4" style="align-items:end">
    <div><div class="label muted small" style="font-weight:700;margin-bottom:6px">NUMBER OF REQUESTS</div>
      <div class="chip-group" id="t-n">${[10, 50, 100, 500].map((n) => `<button class="btn btn-outline ${n === 100 ? "active" : ""}" data-n="${n}">${n}</button>`).join("")}
      <input type="number" id="t-custom" placeholder="Custom" min="1" max="2000" style="width:90px"></div></div>
    <label>ALGORITHM<select id="t-algo">${Object.entries(ALGO_LABEL).map(([k, v]) => `<option value="${k}" ${k === cfg.algorithm ? "selected" : ""}>${v}</option>`).join("")}</select></label>
    <label>SCENARIO<select id="t-scen"><option value="normal">Normal (all healthy)</option><option value="slow">Slow server</option><option value="overload">Overload</option><option value="failure">Failure</option><option value="as_is">Keep current Server Lab state</option></select></label>
    <div class="row"><label>TARGET<select id="t-target">${SERVERS.map((s) => `<option value="${s}">${SRV_LABEL(s)}</option>`).join("")}</select></label>
      <label>RATE (req/s)<input id="t-rate" type="number" value="20" min="0" max="200" style="width:80px"></label>
      <label>CONCURRENCY<input id="t-conc" type="number" value="10" min="1" max="64" style="width:80px"></label></div>
  </div>
  <div class="row" style="margin-top:14px"><button class="btn btn-primary" id="t-start">▶ START TRAFFIC</button><button class="btn btn-danger" id="t-stop" disabled>■ STOP</button>
    <button class="btn btn-outline" id="t-reset">⟲ RESET</button><span class="spacer"></span><span id="t-state" class="badge">idle</span></div></div>

  <div class="flow-wrap"><svg class="flow-svg" id="flow" viewBox="0 0 1000 430"></svg>
    <div class="flow-legend"><span><span class="dot" style="color:#60a5fa"></span> request</span><span><span class="dot" style="color:#4ade80"></span> success</span><span><span class="dot" style="color:#f87171"></span> failure</span></div></div>

  <div class="grid g4" style="margin-top:16px" id="t-kpis"></div>
  <div class="grid g3" style="margin-top:16px">
    <div class="card"><h3>Server distribution (actual)</h3><div id="t-dist"></div></div>
    <div class="card"><h3>Latest DQN Q-values</h3><div id="t-q"></div></div>
    <div class="card"><h3>Latency per request (ms)</h3><div id="t-spark"></div><div class="muted small" id="t-spark-l"></div></div>
  </div>
  <div class="grid g2" style="margin-top:16px">
    <div class="card"><h3>Live servers</h3><div class="grid g2" id="t-servers"></div></div>
    <div class="card"><h3>Request log (actual backend responses)</h3><div class="event-log" id="t-log"></div></div>
  </div>`;

  /* ----- flow diagram ----- */
  const svg = $("#flow");
  const N = { client: [70, 215], mycamu: [260, 215], lb: [470, 215] };
  const SY = { server1: 60, server2: 165, server3: 270, server4: 375 };
  const node = (id, x, y, w, h, title, sub) => `<g class="flow-node" id="n-${id}"><rect x="${x - w / 2}" y="${y - h / 2}" width="${w}" height="${h}" rx="12"/>
    <text x="${x}" y="${y - 2}">${title}</text><text class="sub" x="${x}" y="${y + 16}" id="n-${id}-sub">${sub}</text></g>`;
  svg.innerHTML = `
    ${SERVERS.map((s) => `<path class="flow-link" d="M${N.lb[0] + 70},${N.lb[1]} C 620,${N.lb[1]} 640,${SY[s]} ${800 - 75},${SY[s]}"/>`).join("")}
    <path class="flow-link" d="M${N.client[0] + 55},215 L${N.mycamu[0] - 60},215"/><path class="flow-link" d="M${N.mycamu[0] + 60},215 L${N.lb[0] - 70},215"/>
    ${node("client", ...N.client, 110, 64, "STUDENTS", "requests")}
    ${node("mycamu", ...N.mycamu, 120, 64, "MYCAMU", ":8080")}
    ${node("lb", ...N.lb, 140, 74, "LOAD BALANCER", "🧠 " + ALGO_LABEL[cfg.algorithm])}
    ${SERVERS.map((s) => node(s, 800, SY[s], 150, 70, SRV_LABEL(s), "0 req · –")).join("")}
    <g id="particles"></g>`;
  const particles = $("#particles", svg);
  const live = [];
  const pathPoint = (srv, t) => {  // t in [0,1] along client->mycamu->lb->server
    const pts = [[N.client[0] + 55, 215], [N.mycamu[0], 215], [N.lb[0], 215], [620, 215], [800 - 75, SY[srv]]];
    const seg = Math.min(pts.length - 2, Math.floor(t * (pts.length - 1))); const lt = t * (pts.length - 1) - seg;
    return [pts[seg][0] + (pts[seg + 1][0] - pts[seg][0]) * lt, pts[seg][1] + (pts[seg + 1][1] - pts[seg][1]) * lt];
  };
  function spawn(ev, delay) {
    if (!ev.selected_server || live.length > 140) return;
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("r", 6); c.setAttribute("fill", "#60a5fa"); c.setAttribute("opacity", 0);
    particles.appendChild(c); live.push({ c, ev, born: performance.now() + delay });
  }
  let raf;
  function animate(now) {
    for (let i = live.length - 1; i >= 0; i--) {
      const p = live[i]; const age = now - p.born;
      if (age < 0) continue;
      const fwd = 900, back = 600;
      if (age < fwd) { const [x, y] = pathPoint(p.ev.selected_server, age / fwd); p.c.setAttribute("cx", x); p.c.setAttribute("cy", y); p.c.setAttribute("opacity", 1); }
      else if (age < fwd + back) {
        const t = (age - fwd) / back; p.c.setAttribute("fill", p.ev.success ? "#4ade80" : "#f87171");
        const [x, y] = pathPoint(p.ev.selected_server, 1 - t); p.c.setAttribute("cx", x); p.c.setAttribute("cy", y); p.c.setAttribute("opacity", 1 - t * 0.7);
      } else { p.c.remove(); live.splice(i, 1); }
    }
    raf = requestAnimationFrame(animate);
  }
  raf = requestAnimationFrame(animate);
  S.timers.push(() => cancelAnimationFrame(raf));

  /* ----- controls ----- */
  $$("#t-n [data-n]").forEach((b) => b.addEventListener("click", () => { cfg.n = +b.dataset.n; $("#t-custom").value = ""; $$("#t-n [data-n]").forEach((x) => x.classList.toggle("active", x === b)); }));
  $("#t-custom").addEventListener("input", (e) => { if (+e.target.value > 0) { cfg.n = +e.target.value; $$("#t-n [data-n]").forEach((x) => x.classList.remove("active")); } });
  $("#t-algo").addEventListener("change", (e) => { $("#n-lb-sub").textContent = "🧠 " + ALGO_LABEL[e.target.value]; });

  let since = 0, latencies = [], polling = null;
  const logEl = $("#t-log");
  const startPolling = () => { if (polling) polling(); polling = every(400, poll); };

  $("#t-start").addEventListener("click", async () => {
    const body = { requests: cfg.n, algorithm: $("#t-algo").value, scenario: $("#t-scen").value, target_server: $("#t-target").value,
      rate: +$("#t-rate").value, concurrency: +$("#t-conc").value };
    try {
      await api("/api/traffic/reset", { method: "POST", allowError: true });
      await api("/api/traffic/start", { method: "POST", body });
      since = 0; latencies = []; logEl.innerHTML = ""; $("#n-lb-sub").textContent = "🧠 " + ALGO_LABEL[body.algorithm];
      toast(`Started ${body.requests} real requests via ${ALGO_LABEL[body.algorithm]}`, "ok"); startPolling();
    } catch (e) { toast(e.message, "bad"); }
  });
  $("#t-stop").addEventListener("click", async () => { await api("/api/traffic/stop", { method: "POST" }); toast("Stopping…", "warn"); });
  $("#t-reset").addEventListener("click", async () => {
    try { await api("/api/traffic/reset", { method: "POST" }); since = 0; latencies = []; logEl.innerHTML = ""; renderStats({ state: "idle" }); toast("Reset", "ok"); }
    catch (e) { toast(e.message, "bad"); }
  });

  function renderStats(st) {
    const s = st.summary || {};
    $("#t-state").className = `badge ${st.state === "running" ? "info" : st.state === "completed" ? "ok" : ""}`;
    $("#t-state").textContent = st.state || "idle";
    $("#t-start").disabled = ["starting", "running"].includes(st.state); $("#t-stop").disabled = !["starting", "running"].includes(st.state);
    const k = (label, v) => `<div class="card kpi"><div class="value">${v}</div><div class="label">${label}</div></div>`;
    $("#t-kpis").innerHTML = k("Requests sent", st.sent ?? 0) + k("Completed", st.completed ?? 0) + k("Successful", s.successful ?? 0) + k("Failed", s.failed ?? 0)
      + k("Success rate", pct(s.success_rate)) + k("Avg latency", s.avg_latency_ms != null ? fmt(s.avg_latency_ms) + " ms" : "–")
      + k("Throughput", s.throughput_rps != null ? fmt(s.throughput_rps, 2) + " rps" : "–")
      + k("Routing accuracy", s.routing_accuracy != null ? pct(s.routing_accuracy) : "n/a");
    $("#t-dist").innerHTML = s.server_distribution ? bars(s.server_distribution, { digits: 0, unit: " req", colors: { server1: "#2563eb", server2: "#d97706", server3: "#059669", server4: "#db2777" } })
      + (st.degraded_servers && st.degraded_servers.length ? `<div class="small muted" style="margin-top:8px">Degraded by scenario: <b>${st.degraded_servers.map(SRV_LABEL).join(", ")}</b></div>` : "") : `<div class="muted small">No data yet</div>`;
    $("#t-q").innerHTML = qBars(st.latest_q_values);
    $("#t-spark").innerHTML = sparkline(latencies.slice(-200));
    $("#t-spark-l").textContent = latencies.length ? `last ${Math.min(200, latencies.length)} requests · max ${fmt(Math.max(...latencies.slice(-200)), 0)} ms` : "";
    if (s.server_distribution) SERVERS.forEach((x) => { const sub = $(`#n-${x}-sub`); if (sub) sub.textContent = `${s.server_distribution[x]} req`; });
  }

  async function poll() {
    const st = await api(`/api/traffic/status?since=${since}`);
    if (st.state === "idle") { renderStats(st); return; }
    const evs = st.events || [];
    if (evs.length) {
      const t0 = evs[0].t_done;
      evs.forEach((ev) => {
        spawn(ev, Math.min(380, (ev.t_done - t0) * 1000));
        latencies.push(ev.latency_ms);
        const line = document.createElement("div");
        line.innerHTML = `<span class="muted">#${String(ev.id).padStart(3, "0")}</span> ─→ ${esc(ALGO_LABEL[ev.algorithm])} ─→ <span class="srv">${esc(SRV_LABEL(ev.selected_server))}</span> `
          + `<span class="${ev.success ? "ok" : "bad"}">${ev.success ? "✓" : "✕ " + ev.status_code}</span> ${fmt(ev.latency_ms, 0)} ms`
          + (ev.q_values ? ` <span class="muted">Q[${SERVERS.map((x) => fmt(ev.q_values[x], 2)).join(", ")}]</span>` : "");
        logEl.appendChild(line);
      });
      while (logEl.children.length > 400) logEl.firstChild.remove();
      logEl.scrollTop = logEl.scrollHeight;
      since += evs.length;
    }
    renderStats(st);
    if (!["starting", "running"].includes(st.state) && !evs.length) { if (polling) { polling(); polling = null; } }
  }
  every(2000, async () => {
    const d = await api("/api/workers");
    $("#t-servers").innerHTML = d.workers.map((w) => serverCard(w)).join("");
    d.workers.forEach((w) => { const g = $(`#n-${w.name}`); if (g) g.setAttribute("class", `flow-node srv-${w.status}`); });
  });
  const st0 = await api("/api/traffic/status?since=0");
  if (st0.state !== "idle") { startPolling(); } else renderStats(st0);
};

/* ---------------- ALGORITHM COMPARISON ---------------- */
PAGES.compare = async (el) => {
  el.innerHTML = `
  <div class="card"><div class="grid g4" style="align-items:end">
    <label>SCENARIO<select id="c-scen"><option value="normal">Normal</option><option value="slow">Slow server</option><option value="overload">Overload</option><option value="failure">Failure</option></select></label>
    <label>TARGET SERVER<select id="c-target">${SERVERS.map((s) => `<option value="${s}">${SRV_LABEL(s)}</option>`).join("")}</select></label>
    <label>REQUESTS PER ALGORITHM<input id="c-n" type="number" value="100" min="10" max="1000"></label>
    <div class="row"><label>RATE<input id="c-rate" type="number" value="20" min="0" style="width:70px"></label><label>CONC.<input id="c-conc" type="number" value="10" min="1" max="64" style="width:70px"></label></div>
  </div>
  <div class="row" style="margin-top:12px">${Object.entries(ALGO_LABEL).map(([k, v]) => `<label style="flex-direction:row;align-items:center;font-weight:600;color:var(--text)"><input type="checkbox" class="c-algo" value="${k}" checked> ${v}</label>`).join("")}
    <span class="spacer"></span><button class="btn btn-primary" id="c-run">▶ RUN COMPARISON</button><button class="btn btn-outline" id="c-stop">Stop</button></div>
  <div class="muted small" style="margin-top:8px">Each algorithm gets the same scenario and the same number of real requests. Backends are reset and the scenario re-applied before every run.</div></div>
  <div id="c-progress" style="margin-top:16px"></div>
  <div id="c-results"></div>
  <div class="card" style="margin-top:16px"><h3>Previous runs</h3><div id="c-history"></div></div>`;
  const loadHistory = async () => {
    const h = await api("/api/compare/history");
    $("#c-history").innerHTML = h.runs.length ? `<table><tr><th>When</th><th>Scenario</th><th>Requests</th><th>Avg latency (ms)</th><th>Success rate</th><th></th></tr>
      ${h.runs.map((r, i) => `<tr><td class="small muted">${esc(r.created_at)}</td><td>${esc(r.config.scenario)} (${esc(r.config.target_server)})</td><td>${r.config.requests}</td>
      <td class="small">${Object.entries(r.results).map(([a, x]) => `${ALGO_LABEL[a]}: <b>${fmt(x.avg_latency_ms)}</b>`).join(" · ")}</td>
      <td class="small">${Object.entries(r.results).map(([a, x]) => `${ALGO_LABEL[a]}: ${pct(x.success_rate)}`).join(" · ")}</td>
      <td><button class="btn btn-sm btn-outline" data-h="${i}">View</button></td></tr>`).join("")}</table>` : `<div class="empty">No comparisons yet.</div>`;
    $$("[data-h]", el).forEach((b) => b.addEventListener("click", () => renderResults(h.runs[+b.dataset.h].results, h.runs[+b.dataset.h].config)));
  };
  function renderResults(R, conf) {
    const A = Object.keys(R);
    const pick = (k, mul = 1) => Object.fromEntries(A.map((a) => [a, R[a][k] != null ? R[a][k] * mul : null]));
    const winner = (k, lower) => A.reduce((b, a) => (R[a][k] != null && (b === null || (lower ? R[a][k] < R[b][k] : R[a][k] > R[b][k]))) ? a : b, null);
    const metrics = [["avg_latency_ms", "Avg latency", true], ["p95_latency_ms", "P95 latency", true], ["success_rate", "Success rate", false], ["throughput_rps", "Throughput", false]];
    if (A.some((a) => R[a].routing_accuracy != null)) metrics.push(["routing_accuracy", "Routing accuracy", false]);
    const wins = Object.fromEntries(A.map((a) => [a, 0])); metrics.forEach(([k, , lo]) => { const w = winner(k, lo); if (w) wins[w]++; });
    $("#c-results").innerHTML = `
    <div class="card" style="margin-top:16px"><div class="card-head"><h3>Results – ${esc(conf.scenario)} scenario (${esc(conf.target_server)}), ${conf.requests} requests each</h3></div>
      <div class="table-wrap"><table><tr><th>Algorithm</th><th>Success</th><th>Failed</th><th>Avg ms</th><th>P50</th><th>P95</th><th>P99</th><th>Throughput</th><th>Jain fairness</th><th>CPU of chosen server</th><th>Routing accuracy</th><th>Distribution s1/s2/s3/s4</th><th>Metric wins</th></tr>
      ${A.map((a) => { const x = R[a]; return `<tr><td><b style="color:${ALGO_COLOR[a]}">${ALGO_LABEL[a]}</b></td><td>${pct(x.success_rate)}</td><td>${x.failed}</td><td>${fmt(x.avg_latency_ms)}</td><td>${fmt(x.p50_latency_ms)}</td>
        <td>${fmt(x.p95_latency_ms)}</td><td>${fmt(x.p99_latency_ms)}</td><td>${fmt(x.throughput_rps, 2)}</td><td>${fmt(x.jain_fairness, 3)}</td><td>${fmt(x.avg_cpu_of_selected_server)}%</td>
        <td>${x.routing_accuracy != null ? pct(x.routing_accuracy) : "n/a"}</td><td>${SERVERS.map((s) => x.server_distribution[s]).join(" / ")}</td><td><b>${wins[a]}</b>/${metrics.length}</td></tr>`; }).join("")}</table></div>
      <div class="muted small" style="margin-top:6px">Measured values only. Whichever algorithm is best on a metric gets ★; DQN is not favoured by the UI.</div></div>
    <div class="grid g2" style="margin-top:16px">
      <div class="card"><h3>Average latency (ms) – lower is better</h3>${bars(pick("avg_latency_ms"), { lowerBetter: true, unit: " ms" })}</div>
      <div class="card"><h3>P95 latency (ms) – lower is better</h3>${bars(pick("p95_latency_ms"), { lowerBetter: true, unit: " ms" })}</div>
      <div class="card"><h3>Success rate (%)</h3>${bars(pick("success_rate", 100), { unit: "%" })}</div>
      <div class="card"><h3>Throughput (successful req/s)</h3>${bars(pick("throughput_rps"), { digits: 2 })}</div>
      ${A.some((a) => R[a].routing_accuracy != null) ? `<div class="card"><h3>Routing accuracy – avoided the degraded server (%)</h3>${bars(pick("routing_accuracy", 100), { unit: "%" })}</div>` : ""}
      <div class="card"><h3>Server distribution</h3>${A.map((a) => `<div class="small" style="margin:6px 0 2px"><b>${ALGO_LABEL[a]}</b></div>${distStrip(R[a].server_distribution)}`).join("")}</div>
    </div>`;
  }
  function distStrip(d) {
    const tot = SERVERS.reduce((s, x) => s + (d[x] || 0), 0) || 1; const col = { server1: "#2563eb", server2: "#d97706", server3: "#059669", server4: "#db2777" };
    return `<div style="display:flex;height:18px;border-radius:6px;overflow:hidden">${SERVERS.map((x) => `<div title="${SRV_LABEL(x)}: ${d[x]}" style="width:${(100 * (d[x] || 0)) / tot}%;background:${col[x]};color:#fff;font-size:10px;text-align:center">${d[x] ? x.slice(-1) : ""}</div>`).join("")}</div>`;
  }
  let pollStop = null;
  const poll = async () => {
    const st = await api("/api/compare/status");
    if (st.state === "idle") return;
    const algos = st.config.algorithms;
    $("#c-progress").innerHTML = `<div class="card"><div class="row"><b>${st.state === "running" ? `<span class="spinner"></span> Running` : st.state}</b><span class="muted">${st.current ? "now: " + ALGO_LABEL[st.current] : ""}</span><span class="spacer"></span><span class="muted small">${st.elapsed_seconds}s</span></div>
      ${algos.map((a) => `<div class="bar-row" style="margin-top:8px"><span>${ALGO_LABEL[a]}</span><div class="bar-track"><div class="bar-fill" style="width:${(100 * (st.progress[a] || 0)) / st.config.requests}%;background:${ALGO_COLOR[a]}"></div></div><span class="bar-val">${st.progress[a] || 0}/${st.config.requests}</span></div>`).join("")}</div>`;
    if (Object.keys(st.results).length) renderResults(st.results, st.config);
    if (st.state !== "running" && pollStop) { pollStop(); pollStop = null; loadHistory(); }
  };
  $("#c-run").addEventListener("click", async () => {
    const algorithms = $$(".c-algo").filter((c) => c.checked).map((c) => c.value);
    try {
      await api("/api/compare/start", { method: "POST", body: { scenario: $("#c-scen").value, target_server: $("#c-target").value, requests: +$("#c-n").value, algorithms, rate: +$("#c-rate").value, concurrency: +$("#c-conc").value } });
      toast("Comparison started", "ok"); if (pollStop) pollStop(); pollStop = every(800, poll);
    } catch (e) { toast(e.message, "bad"); }
  });
  $("#c-stop").addEventListener("click", () => api("/api/compare/stop", { method: "POST" }).then(() => toast("Stopping after current algorithm", "warn")));
  loadHistory();
  const st = await api("/api/compare/status");
  if (st.state === "running") pollStop = every(800, poll); else if (st.state !== "idle") poll();
};

/* ---------------- SYSTEM HEALTH ---------------- */
PAGES.health = async (el) => {
  const NAMES = { round_robin: "Round Robin", least_connections: "Least Connections", aco: "ACO", dqn: "DQN", server1: "Server 1", server2: "Server 2", server3: "Server 3", server4: "Server 4" };
  el.innerHTML = `<div class="row" style="margin-bottom:12px"><span class="muted">Every tile is the result of a real <span class="mono">/health</span> call made by <span class="mono">GET /api/health</span> (refresh 5 s).</span><span class="spacer"></span><span id="h-time" class="muted small"></span></div>
    <div class="grid g3" id="h-grid"></div>
    <div class="card" style="margin-top:16px"><h3>Monitoring</h3><div class="row"><a class="btn btn-outline" href="http://localhost:9090" target="_blank" rel="noopener">Prometheus :9090 ↗</a>
    <a class="btn btn-outline" href="http://localhost:3000" target="_blank" rel="noopener">Grafana :3000 ↗</a><span class="muted small">MyCamu exposes its own metrics at <span class="mono">/metrics</span>.</span></div></div>`;
  every(5000, async () => {
    const d = await api("/api/health");
    const tile = (name, ok, sub) => `<div class="card health-tile"><span class="big">${ok ? "🟢" : "🔴"}</span><div><b>${esc(name)}</b><div class="small ${ok ? "" : "muted"}">${ok ? "Healthy" : "Unavailable"} ${sub ? "· " + sub : ""}</div></div></div>`;
    $("#h-grid").innerHTML = tile("MyCamu", d.success, d.database ? "database OK" : "database error")
      + Object.entries(d.services).map(([k, ok]) => { const det = d.details[k] || {}; return tile(NAMES[k] || k, ok, det.reachable ? `HTTP ${det.status_code} · ${det.response_ms} ms` : esc(det.error || "")); }).join("");
    $("#h-time").textContent = "checked " + new Date(d.timestamp * 1000).toLocaleTimeString() + (d.all_healthy ? " · all healthy" : " · some services unavailable");
  });
};

/* ---------------- boot ---------------- */
(async function boot() {
  try {
    const d = await api("/api/auth/me", { allow401: true, allowError: true });
    if (d.success) { S.student = d.student; showApp(); } else showLogin();
  } catch (_) { showLogin(); }
})();
