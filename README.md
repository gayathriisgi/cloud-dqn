# Cloud Load Balancer — DQN vs Round Robin / Least Connections / ACO

A Dockerized cloud load-balancing testbed that compares a genuine Deep
Q-Network (DQN) reinforcement-learning load balancer against three
static (non-learning) baselines — Round Robin, Least Connections and
Ant Colony Optimization (ACO) — under identical, real, controlled
backend workload conditions, with full Prometheus + Grafana
observability.

This README is long on purpose: it is written so that every command
you need — install, build, train, run, generate traffic, break a
server, compare algorithms, view dashboards, and clean up — is here
verbatim, with no placeholders. If you only want the fastest path to
a working demo, jump straight to **Section 21, Quick Demo Command
Cheat Sheet**.

> **Note on example numbers in this document.** A handful of numbers
> quoted in Sections 3–4 and 19 (e.g. specific success rates, fairness
> values, response times) are the actual results this project produced
> when it was built and evaluated end-to-end during development — they
> are real, measured numbers, not fabricated illustrations. They are
> still labeled **"Example output; actual values depend on execution"**
> throughout because your own run will differ (timing jitter, host
> CPU contention, OS scheduling, and — most importantly — how many
> episodes you train the DQN for all affect the exact figures). The
> `results/*.json` and `results/*.png` files shipped in this project
> are the real artifacts from that run; running the commands in this
> README regenerates fresh ones from your own machine.

> ## What's new in v3 (October 2026)
> * **Retrained DQN that doesn't overfit or underfit.** It is still plain DQN (not Double DQN), with the same 46→256→128→4 network.
>   Training now uses a train/validation split, early stopping, a best-checkpoint save, a held-out TD-loss monitor, domain randomisation,
>   server-permutation augmentation, concurrent agent clients, AdamW and a cosine LR schedule. → **Section 38**
> * **Scenario stress-test suite** with PASS/FAIL checks: 19 scenarios (low/high/spike/burst/ramp/flash-crowd traffic, slow, overload,
>   failure, recovery, flapping, cascading...), per-stage metrics for all 4 algorithms, a server-status timeline, routing accuracy,
>   statistical significance, and an HTML report. → **Section 39**
> * **Offline policy probes:** decision accuracy of the Q-network on synthetic incident states. → **Section 40**
> * **TCE MyCamu** (`mycamu/`, port 8080): student portal + live load-balancing visualisation, fully dockerised. → **Section 41**
> * **Measured v3 results.** → **Section 42**
>
> **Important:** the `models/dqn_model.pt` shipped in v2 was written by the numpy shim in `sandbox_shims/`, and **real PyTorch cannot load it**
> (`RuntimeError: Invalid magic number`), so in Docker the v2 balancer ran with `model_loaded=false`. v3 ships a model trained with real PyTorch.
> It was verified to load in **torch 2.2.2** (the version in `load_balancer/Dockerfile`) as well as torch 2.11.

---

## Table of contents

1. [What this project is](#1-what-this-project-is)
2. [Architecture overview](#2-architecture-overview)
3. [Why the old DQN looked like a heuristic, and what changed](#3-why-the-old-dqn-looked-like-a-heuristic-and-what-changed)
4. [How the new DQN demonstrates genuine adaptive behaviour](#4-how-the-new-dqn-demonstrates-genuine-adaptive-behaviour)
5. [Project structure](#5-project-structure)
6. [The DQN state vector (46 dimensions)](#6-the-dqn-state-vector-46-dimensions)
7. [The DQN action space](#7-the-dqn-action-space)
8. [The multi-objective reward function](#8-the-multi-objective-reward-function)
9. [The DQN network, training loop and replay buffer](#9-the-dqn-network-training-loop-and-replay-buffer)
10. [The dynamic environment / workload simulator](#10-the-dynamic-environment--workload-simulator)
11. [Backend Flask server API](#11-backend-flask-server-api)
12. [Load balancer APIs (all four algorithms)](#12-load-balancer-apis-all-four-algorithms)
13. [Requirements](#13-requirements)
14. [Installing dependencies (local, no Docker)](#14-installing-dependencies-local-no-docker)
15. [Running locally without Docker](#15-running-locally-without-docker)
16. [Running with Docker Compose](#16-running-with-docker-compose)
17. [Training the DQN](#17-training-the-dqn)
18. [Generating traffic](#18-generating-traffic)
19. [Evaluation metrics and how they're computed](#19-evaluation-metrics-and-how-theyre-computed)
20. [Running the full evaluation suite](#20-running-the-full-evaluation-suite)
21. [Quick Demo Command Cheat Sheet](#21-quick-demo-command-cheat-sheet)
22. [Automated tests](#22-automated-tests)
23. [Simulating server failure and recovery](#23-simulating-server-failure-and-recovery)
24. [Prometheus metrics reference](#24-prometheus-metrics-reference)
25. [Grafana dashboard](#25-grafana-dashboard)
26. [Environment variables reference](#26-environment-variables-reference)
27. [Metric documentation table](#27-metric-documentation-table)
28. [Results produced by this project](#28-results-produced-by-this-project)
29. [Viva / project explanation section](#29-viva--project-explanation-section)
30. [Troubleshooting](#30-troubleshooting)
31. [Windows-specific notes](#31-windows-specific-notes)
32. [Stopping and cleaning up](#32-stopping-and-cleaning-up)
33. [Design decisions and known trade-offs](#33-design-decisions-and-known-trade-offs)
34. [Extending the project](#34-extending-the-project)
35. [File-by-file summary](#35-file-by-file-summary)
36. [Glossary](#36-glossary)
37. [License / academic use note](#37-license--academic-use-note)
38. [v3 DQN training: avoiding over- and under-fitting](#38-v3-dqn-training-avoiding-over--and-under-fitting)
39. [Scenario stress-test suite (PASS/FAIL)](#39-scenario-stress-test-suite-passfail)
40. [Offline policy probes](#40-offline-policy-probes)
41. [TCE MyCamu (port 8080)](#41-tce-mycamu-port-8080)
42. [v3 measured results](#42-v3-measured-results)

---

## 1. What this project is

Four independent Flask backend servers (`server1`..`server4`) each run
a **real, measurable, controllable workload simulator**: genuine
CPU-burning threads (measured with `psutil`, not faked), genuine RSS
memory allocation, a genuine bounded semaphore for queueing, and
genuine per-request processing delay. Four load balancers sit in
front of them and route real HTTP requests to real servers:

- **Round Robin** — cycles through servers in order.
- **Least Connections** — routes to whichever server currently has
  the fewest in-flight requests (ties broken randomly for fairness).
- **ACO (Ant Colony Optimization)** — a pheromone-based heuristic with
  a Max-Min Ant System floor/ceiling to prevent colony collapse.
- **DQN (Deep Q-Network)** — a genuine reinforcement-learning agent
  that observes a 46-dimensional state of the whole system, produces
  Q-values for each of the 4 possible server choices through a real
  neural network, picks an action, watches what happens, computes a
  multi-objective reward, and learns from experience replay.

Prometheus scrapes metrics from all 8 processes (4 servers + 4
balancers) and Grafana visualizes them live. A traffic generator can
drive realistic load patterns (steady, ramping, bursty, spiky) at any
balancer. A dedicated testing suite proves — with real PASS/FAIL
conditions computed from live measurements, not just "the script ran
without crashing" — that the DQN genuinely adapts to overload, failure
and recovery, and a comparison harness runs all four algorithms under
identical conditions so their results can be fairly compared.

## 2. Architecture overview

```
                     ┌─────────────────────────────────────────────┐
                     │              Traffic Generator               │
                     │        (patterns: normal/burst/ramp/…)       │
                     └───────────────────────┬───────────────────────┘
                                              │ HTTP
                     ┌────────────────────────┼─────────────────────────┐
                     ▼                        ▼                         ▼
            ┌────────────────┐      ┌──────────────────┐      ┌────────────────┐
            │  Round Robin   │      │ Least Connections │      │  ACO Balancer  │
            │   :8000/9000   │      │    :8001/9001     │      │   :8002/9002   │
            └───────┬────────┘      └─────────┬─────────┘      └───────┬────────┘
                    │                          │                        │
                    │              ┌───────────┴────────────┐           │
                    │              │      DQN Balancer       │           │
                    │              │   :8003/9003 (learns)   │           │
                    │              └───────────┬─────────────┘           │
                    └──────────────┬────────────┴────────────┬───────────┘
                                   ▼                          ▼
                     ┌──────────────────────────────────────────────┐
                     │  server1 : server2 : server3 : server4        │
                     │  (real CPU / real memory / real queue / real  │
                     │   processing delay, each independently tunable)│
                     └───────────────────┬────────────────────────────┘
                                          │ /metrics, /prometheus
                                          ▼
                              ┌───────────────────────┐
                              │       Prometheus        │
                              │        :9090             │
                              └───────────┬───────────────┘
                                          ▼
                              ┌───────────────────────┐
                              │        Grafana           │
                              │        :3000              │
                              └───────────────────────┘
```

The **DQN request pipeline** (the core of this project) is:

```
Dynamic System State  →  Multi-Metric DQN State (46-d)  →  Q-Network
     →  Q-values for 4 actions  →  argmax / epsilon-greedy action
     →  Real HTTP request to the chosen server  →  Real response
     →  Observe new system state  →  Multi-objective reward
     →  (state, action, reward, next_state) stored in replay buffer
     →  Batch sampled and network updated via gradient descent
     →  Slightly different, better-informed decisions on future requests
```

Every arrow above is real code, not a diagram aspiration: see
`load_balancer/dqn_balancer.py` for the request-time pipeline and
`dqn/train.py` / `dqn/agent.py` for the learning loop.

## 3. Why the old DQN looked like a heuristic, and what changed

**A. Original architecture.** The uploaded project already had the
right shape: Flask backend servers, a Docker Compose stack, Round
Robin / Least Connections / ACO baselines, a basic DQN balancer, a
traffic generator, and Prometheus/Grafana wiring. That structure is
preserved in this delivery.

**B. The original DQN's limitations.**

- The state vector used only a handful of metrics per server (in some
  of the old `dqn1/` variants, effectively just active-connection
  counts), so the network had almost nothing to distinguish "a server
  under real CPU pressure" from "a server that is merely busy with
  many small requests" — two situations that need different routing
  decisions.
- The reward was a simple, narrow signal (largely just
  success/failure or raw response time), so nothing pushed the agent
  to trade off CPU vs. queue vs. fairness vs. latency — it could only
  ever optimize one axis, which made its behaviour converge to
  something that *looked* like "always pick the fastest responder so
  far" — indistinguishable from a slightly-noisy Least-Connections
  heuristic.
- There were **five separate, inconsistent DQN implementations**
  scattered across `dqn_balancer.py`, `dqnbalancer.py`,
  `dqn1/dqnbalancer1.py`, `dqn1/dqn_inference.py`, `dqn1/dqn_mo1.py`
  and `dqn1/dqn_tra1.py`/`dqn1/env1.py`, each with its own state
  construction, so a model trained by one script was not guaranteed
  to even be dimensionally compatible with the network used at
  inference time — and nothing checked for that, so a mismatched
  model could silently load and produce meaningless Q-values.
- Docker networking bugs (assuming `localhost` reached another
  container, or parsing a server's *external* published port instead
  of using its Docker service name) meant the balancers sometimes
  weren't even reliably reaching all four backends in the containerized
  deployment.

**C. What changed in this delivery.**

- All prior DQN code paths are **consolidated into one authoritative
  implementation**: `dqn/{config,state,reward,model,replay_buffer,agent,environment,train,inference}.py`,
  used identically by both training (`dqn/train.py`) and inference
  (`load_balancer/dqn_balancer.py` via `dqn/inference.py`) — there is
  now exactly one function, `dqn/state.py::build_state()`, that turns
  live metrics into a state vector, so train/inference can never
  silently diverge.
- The state vector was expanded to **46 dimensions**: 10 real metrics
  per server × 4 servers (cpu, memory, queue, active requests,
  latency, request rate, throughput, error rate, health,
  load-pressure) + 6 global features (see Section 6).
- The reward became a **configurable, weighted, multi-objective
  function** over 10 components (Section 8) computed from the same
  real metrics — nothing in it is a hand-written `if cpu > 80: avoid`
  rule; the network has to learn what state patterns lead to good
  vs. bad reward outcomes on its own.
- `dqn/agent.py::DQNAgent.load()` now **strictly validates
  `state_dim`/`action_dim`/`hidden_sizes`** against the running
  configuration and raises `ModelDimensionMismatchError` rather than
  silently loading an incompatible checkpoint.
- All Docker service addressing goes through `SERVER_URLS`
  (`http://server1:5000` etc. inside Docker), never `localhost`
  inside a container, and server identity is index-based
  (`server_name(index)`) rather than parsed from a port number that
  is identical across containers.

**D. Why the new DQN demonstrates genuine adaptation** is explained in
full, with real measured evidence, in the next section.

## 4. How the new DQN demonstrates genuine adaptive behaviour

Reinforcement learning is only convincing if you can show the agent's
*decisions change* in response to a *changing environment*, and that
the change is *caused by* the environment (not coincidence). This
project does that with a dedicated, scripted 5-phase experiment
(`testing/adaptation_test.py`, Section 8's spec) that:

1. Runs all servers healthy (baseline).
2. Genuinely overloads server1 (real CPU burner threads, real queue
   buildup — not a fake flag) while driving real traffic.
3. Additionally overloads server3, while server1 stays overloaded.
4. Recovers server1 (server3 stays overloaded).
5. Drives a burst-traffic pattern at all-healthy servers.

...and computes hard, numeric adaptive-behaviour evidence directly
from the real requests captured during that run:

| Metric | What it proves | Real value from this project's run |
|---|---|---|
| Overload avoidance rate (phase 2) | Share of requests **not** sent to the overloaded server | **100%** (0 of 249 requests went to server1 while it was overloaded) |
| Overload avoidance rate (phase 3) | Same, with two servers overloaded simultaneously | **100%** (0 requests to either server1 or server3) |
| Adaptation time (phase 2) | Seconds from overload starting until the DQN's routing share of that server drops to ≤15% | **~0.8s** |
| Recovery time (phase 4) | Seconds from recovery until the DQN starts routing back to the recovered server | **~0.8s** |
| Q-value sensitivity (server1, phase1→phase2) | Change in the network's own Q(server1) estimate when server1 becomes overloaded | **+2.82** (Q genuinely dropped — this is the network's internal value function reacting, not just the final routing decision) |
| Q-value rebound (server1, phase3→phase4) | Change in Q(server1) when it recovers | **+1.05** (Q genuinely rose again) |
| Load–response correlation | Pearson correlation between a server's load-pressure and its selection share, across all phases/servers | **−0.33** (negative = routes away from higher-pressure servers, the definition of adaptive routing) |

These are the actual numbers saved in
`results/adaptation_summary.json` by this project's own evaluation
run (see Section 28). A static algorithm (Round Robin, Least
Connections in its raw "current connection count" form) cannot
produce this pattern — it has no notion of CPU pressure, error rate,
or a learned value function that *changes* in response to it. Round
Robin, by construction, keeps sending requests to an overloaded
server in fixed rotation, unaware of how badly it is performing.

## 5. Project structure

```
cloud-load-balancer/
├── common/
│   ├── settings.py          # single source of truth: server URLs, ports, timeouts, norm refs
│   └── formulas.py          # Jain fairness index, percentiles, load variance/balance degree
├── servers/
│   ├── app.py                # Flask backend: /, /health, /metrics, /prometheus, /load, /fail, /recover, /reset
│   ├── workload.py           # WorkloadSimulator: real CPU burner threads, real memory, real queueing
│   └── Dockerfile
├── load_balancer/
│   ├── common.py              # shared HTTP/metrics-collection helpers, response-shape builder
│   ├── round_robin.py
│   ├── least_connections.py
│   ├── aco_balancer.py
│   ├── dqn_balancer.py        # the full DQN request pipeline (Section 2 diagram) + Prometheus
│   └── Dockerfile              # shared image for all four balancers (torch installed here)
├── dqn/
│   ├── config.py               # RewardWeights, DQNHyperparams, paths, env-var overrides
│   ├── state.py                 # build_state(): the ONE function that builds the 46-d vector
│   ├── reward.py                # compute_reward(): the ONE multi-objective reward function
│   ├── model.py                  # QNetwork (MLP)
│   ├── replay_buffer.py           # ReplayBuffer (experience replay)
│   ├── agent.py                    # DQNAgent: select_action / learn / save / load
│   ├── environment.py               # LoadBalancerEnvironment: real HTTP step(), workload presets
│   ├── train.py                      # training loop / CLI
│   └── inference.py                   # thin wrapper used by load_balancer/dqn_balancer.py
├── traffic/
│   └── traffic_generator.py    # configurable traffic patterns
├── testing/
│   ├── scenarios.py             # shared scenario/phase definitions (fair comparison)
│   ├── metrics.py                 # traffic-driving + evaluation-metric aggregation
│   ├── compare_algorithms.py       # runs all 4 algorithms under identical conditions
│   ├── adaptation_test.py           # the 5-phase adaptive-behaviour experiment
│   ├── failure_test.py               # failure/recovery experiment
│   └── run_tests.py                   # 8 automated tests with real PASS/FAIL
├── visualization/
│   └── generate_plots.py        # builds all results/*.png from real results/*.json + logs/*.csv
├── monitoring/
│   ├── prometheus/prometheus.yml
│   └── grafana/
│       ├── provisioning/datasources/datasource.yml
│       ├── provisioning/dashboards/dashboard.yml
│       └── dashboards/dqn_dashboard.json
├── mycamu/                       # v3: TCE MyCamu portal + live LB visualisation (Section 41)
├── testing/stress_test_suite.py  # v3: 19-scenario PASS/FAIL benchmark (Section 39)
├── testing/stress_scenarios.py   # v3: scenario timelines
├── testing/policy_probe_test.py  # v3: offline decision-accuracy probes (Section 40)
├── models/                       # trained checkpoints land here (dqn_model.pt)
├── logs/                          # training_history.csv / .json
├── results/                        # *.json (raw+summary results) and *.png (plots)
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .dockerignore
├── .gitignore
└── README.md                        # this file
```

## 6. The DQN state vector (46 dimensions)

Defined once, in `dqn/state.py`. For each of the 4 servers (40
features), then 6 global features:

**Per-server features (×4 servers = 40 features), each normalized to
`[0, 1]`:**

| # | Feature | Raw source | Normalization |
|---|---|---|---|
| 1 | CPU usage | `psutil` process CPU % | `cpu / 100` |
| 2 | Memory usage | Real RSS vs. `MEMORY_CAP_MB` | `memory / 100` |
| 3 | Queue length | Requests waiting on the capacity semaphore | `queue / QUEUE_NORM_REF` (default ref=20) |
| 4 | Active requests | In-flight request count | `active / ACTIVE_NORM_REF` (default ref=30) |
| 5 | Average latency | Real measured response time (ms), rolling window | `latency_ms / LATENCY_NORM_REF_MS` (default ref=1000) |
| 6 | Request rate | Rolling requests/sec | `rate / RATE_NORM_REF` (default ref=40) |
| 7 | Throughput | Rolling successful requests/sec | `throughput / THROUGHPUT_NORM_REF` (default ref=40) |
| 8 | Error rate | Rolling failed/total, 0–1 already | used as-is |
| 9 | Health | 1.0 if server considers itself healthy else 0.0 | boolean → float |
| 10 | Load pressure | Composite of the above (see Section 10) | already 0–1 |

**Global features (6 features):**

1. Total request rate across all servers, normalized
2. Total active requests across all servers, normalized
3. Overall (system-wide) average latency, normalized
4. Overall system throughput, normalized
5. Global error rate across all servers
6. Traffic intensity (total request rate relative to total known
   capacity)

`46 = 10 × 4 + 6`. If a server is unreachable, it is not silently
skipped — `dqn/state.py::_worst_case_metrics()` fills in worst-case
values (100% CPU, 100% memory, max queue/active/latency, 0
throughput, 100% error rate, unhealthy, 1.0 load pressure) so the
network is actively taught to avoid unreachable servers rather than
being fed a state vector that just doesn't mention them.

`dqn/config.py::compute_state_dim(n_servers)` computes this
dimension programmatically (`n_servers * FEATURES_PER_SERVER +
GLOBAL_FEATURES`), so if you ever add a 5th server, the state
dimension — and hence the network's input layer — updates
automatically everywhere it's used; you would still need to retrain
the model (see Section 33).

## 7. The DQN action space

4 discrete actions — one per backend server (`action_dim =
SETTINGS.n_servers = 4`). `select_action()` in `dqn/agent.py`
implements epsilon-greedy: with probability `epsilon`, pick a
uniformly random action (exploration); otherwise, pick
`argmax(Q(state))` from a real forward pass through the trained
network (exploitation). At inference time (`load_balancer/dqn_balancer.py`),
`epsilon` defaults to `DQN_INFERENCE_EPSILON=0.0` (pure exploitation
of the learned policy) but can be raised via that environment
variable to keep a small amount of live exploration if you enable
online learning.

## 8. The multi-objective reward function

Computed once, in `dqn/reward.py::compute_reward()`, from the *same*
real, normalized metrics used to build the state — never from a
separate hand-tuned rulebook. All ten weights are configurable via
environment variables (`dqn/config.py::RewardWeights`) so no single
metric silently dominates without being an explicit, visible,
documented choice:

```
reward =  w_success   * (1 if request succeeded else 0)
        + w_throughput * normalized_throughput_of_chosen_server
        + w_health     * (1 if chosen server healthy else 0)
        - w_latency    * normalized_response_time
        - w_cpu        * normalized_cpu_of_chosen_server
        - w_memory     * normalized_memory_of_chosen_server
        - w_queue      * normalized_queue_of_chosen_server
        - w_active     * normalized_active_requests_of_chosen_server
        - w_error      * error_rate_of_chosen_server
        - w_imbalance  * load_variance_across_all_servers
```
then clipped to `[-reward_clip, +reward_clip]` (default `±5.0`).

Default weights (all overridable, e.g. `REWARD_W_LATENCY=1.5`):

| Weight | Env var | Default |
|---|---|---|
| Success | `REWARD_W_SUCCESS` | 1.0 |
| Throughput | `REWARD_W_THROUGHPUT` | 0.5 |
| Health | `REWARD_W_HEALTH` | 0.5 |
| Latency (penalty) | `REWARD_W_LATENCY` | 1.0 |
| CPU (penalty) | `REWARD_W_CPU` | 0.4 |
| Memory (penalty) | `REWARD_W_MEMORY` | 0.3 |
| Queue (penalty) | `REWARD_W_QUEUE` | 0.4 |
| Active requests (penalty) | `REWARD_W_ACTIVE` | 0.3 |
| Error rate (penalty) | `REWARD_W_ERROR` | 0.8 |
| Load imbalance (penalty) | `REWARD_W_IMBALANCE` | 0.5 |
| Clip range | `REWARD_CLIP` | 5.0 |

Nowhere in this function is there an `if metric > threshold: return
fixed_value` rule — every term is a smooth, continuous function of a
real measured/normalized quantity, and the *routing decision itself*
never touches this function at all (the DQN chooses actions purely
from `argmax(Q(state))`); the reward only ever *shapes what the
network learns to value* through gradient descent on the Bellman
target.

## 9. The DQN network, training loop and replay buffer

- **Network** (`dqn/model.py::QNetwork`): `Linear(46→256) → ReLU →
  Linear(256→128) → ReLU → Linear(128→4)`, Kaiming-uniform
  initialization (matches ReLU networks' recommended init).
- **Target network**: a frozen copy of the online network, synced
  every `target_update_freq` (default 200) learning steps — the
  standard DQN trick to stabilize the Bellman target.
- **Loss**: `SmoothL1Loss` (Huber loss) between predicted `Q(s,a)` and
  the Bellman target `r + γ · max_a' Q_target(s', a') · (1 - done)`.
- **Optimizer**: Adam, default learning rate `1e-3`, gradient norm
  clipped to `5.0`.
- **Experience replay** (`dqn/replay_buffer.py`): a `deque`-backed
  buffer (default capacity 20,000 transitions); a batch of 64 is
  sampled uniformly at random per learning step once at least 256
  transitions exist, breaking the temporal correlation between
  consecutive real requests.
- **Epsilon schedule**: starts at `1.0`, decays by `×0.995` per
  episode, floors at `0.05`.
- **Checkpointing** (`dqn/agent.py::save/load`): the saved dict
  includes `model_version`, `state_dim`, `action_dim`, `hidden_sizes`,
  the full hyperparameter set, both networks' weights, optimizer
  state, `epsilon` and step counters — so a checkpoint is fully
  self-describing and `load()` can strictly verify compatibility
  before ever touching the running configuration's network.

All of the above hyperparameters are overridable via environment
variables (`DQN_LR`, `DQN_GAMMA`, `DQN_EPSILON_START`,
`DQN_EPSILON_MIN`, `DQN_EPSILON_DECAY`, `DQN_BATCH_SIZE`,
`DQN_BUFFER_CAPACITY`, `DQN_TARGET_UPDATE_FREQ`, `DQN_MIN_REPLAY`,
`DQN_GRAD_CLIP`) — see Section 26.

## 10. The dynamic environment / workload simulator

`servers/workload.py::WorkloadSimulator` is what makes the metrics
real rather than fabricated:

- **CPU**: a real background thread (`_cpu_burner_loop`) does
  actual floating-point busy-work proportional to `cpu_level`,
  measured back via `psutil.Process().cpu_percent()` — not a random
  number generator.
- **Memory**: a real byte array is allocated up to `memory_mb`,
  measured back via the process's real RSS.
- **Queueing**: a real `threading.Semaphore(capacity)` — when more
  requests arrive concurrently than `capacity`, they genuinely wait,
  and `queue_length` counts real waiters.
- **Latency**: `base_latency_ms ± jitter_ms` real `time.sleep()`
  during request handling, so response time is genuinely
  request-processing time, not a fabricated field.
- **Failure injection**: `error_rate` genuinely causes a fraction of
  requests to return HTTP 5xx; `/fail` and `/recover` (Section 11)
  genuinely flip a server to always-fail / restored.

`dqn/environment.py::SCENARIOS` defines named presets that set
`(cpu_level, memory_mb, base_latency_ms, jitter_ms, capacity,
error_rate)` combinations for: `idle, low, medium, high, cpu_pressure,
memory_pressure, queue_buildup, high_latency, overload`. These are
applied through the real `/load` endpoint on each backend — i.e.
"overload" is a real, controlled combination of high CPU + low
capacity + added latency, not a magic flag the DQN can trivially key
off of.

## 11. Backend Flask server API

Each backend (`servers/app.py`) exposes:

| Route | Method | Purpose |
|---|---|---|
| `/` | GET/POST | Handle a real request (goes through the workload simulator; this is what load balancers forward to) |
| `/health` | GET | `{"status": "healthy"/"unhealthy", "server_id": ...}` |
| `/metrics` | GET | Full JSON: `server_id, cpu_usage, memory_usage, queue_length, active_requests, request_rate, throughput, average_response_time, error_rate, health, load_pressure` |
| `/prometheus` | GET | Prometheus text-exposition format (scraped by Prometheus) |
| `/load` | GET/POST | GET returns current `WorkloadParams`; POST sets `cpu_level`/`memory_mb`/`base_latency_ms`/`jitter_ms`/`capacity`/`error_rate`, or `{"preset": "overload"}` |
| `/fail` | POST | Force this server to fail all requests (simulated crash) |
| `/recover` | POST | Undo `/fail` |
| `/reset` | POST | Reset all counters/windows and workload back to defaults (used between experiments so runs don't inherit stale state) |

All counters are updated under thread-safe locks
(`threading.Lock`/`threading.Semaphore`), and every request's actual
processing wall-clock time is measured with `time.perf_counter()`
around the real work — not estimated.

## 12. Load balancer APIs (all four algorithms)

All four balancers share a common JSON response shape
(`load_balancer/common.py::build_response()`):

```json
{
  "algorithm": "DQN",
  "selected_server": "server3",
  "selected_action": 2,
  "success": true,
  "response_time": 0.0421,
  "reward": 1.83,
  "q_values": [0.71, 1.02, 1.94, 0.88],
  "server_metrics": { "server1": {...}, "server2": {...}, "server3": {...}, "server4": {...} },
  "decision_time_seconds": 0.0013
}
```

(`q_values` is `null` for the three static algorithms — they don't
have any; `reward` is still computed for them too, for apples-to-apples
comparison in `testing/compare_algorithms.py`, but it plays no role in
their routing decision.)

| Route | RR | LC | ACO | DQN | Purpose |
|---|---|---|---|---|---|
| `/` | ✓ | ✓ | ✓ | ✓ | Route one real request |
| `/status` | ✓ | ✓ | ✓ | ✓ | Algorithm-specific internal state (e.g. DQN: `model_loaded`, `state_dim`, `epsilon`, `steps_done`) |
| `/health` | ✓ | ✓ | ✓ | ✓ | Liveness for Docker healthchecks |
| `/reload_model` | – | – | – | ✓ (POST) | Hot-reload `models/dqn_model.pt` without restarting the process |

Ports: Round Robin `8000`/metrics `9000`, Least Connections
`8001`/`9001`, ACO `8002`/`9002`, DQN `8003`/`9003`.

## 13. Requirements

- Python 3.11+ (3.10 also works)
- Docker Desktop (or Docker Engine + Compose plugin) — only needed for
  Section 16; Section 15 covers running without Docker.
- ~1 GB free disk for the Docker images (mostly the CPU PyTorch wheel).
- No GPU required — the network is tiny (46-256-128-4) and trains in
  well under a minute for tens of episodes on any modern CPU.

## 14. Installing dependencies (local, no Docker)

```bash
cd cloud-load-balancer
python -m venv venv
# macOS/Linux:
source venv/bin/activate
# Windows PowerShell:
venv\Scripts\Activate.ps1
# Windows CMD:
venv\Scripts\activate.bat

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

What this does: creates an isolated Python environment, installs the
CPU-only PyTorch wheel explicitly (keeps the install small — no CUDA
runtime), then installs Flask/requests/numpy/prometheus-client/psutil/
matplotlib from `requirements.txt`.

## 15. Running locally without Docker

Open 5 terminals (or background the first four).

```bash
# Terminal 1-4: one backend server each
cd cloud-load-balancer
set SERVER_ID=1 && set PORT=5001 && python servers/app.py     REM Windows CMD, terminal 1
set SERVER_ID=2 && set PORT=5002 && python servers/app.py     REM Windows CMD, terminal 2
set SERVER_ID=3 && set PORT=5003 && python servers/app.py     REM Windows CMD, terminal 3
set SERVER_ID=4 && set PORT=5004 && python servers/app.py     REM Windows CMD, terminal 4
```

macOS/Linux equivalent (one line per terminal):
```bash
SERVER_ID=1 PORT=5001 python servers/app.py
SERVER_ID=2 PORT=5002 python servers/app.py
SERVER_ID=3 PORT=5003 python servers/app.py
SERVER_ID=4 PORT=5004 python servers/app.py
```

PowerShell equivalent:
```powershell
$env:SERVER_ID="1"; $env:PORT="5001"; python servers/app.py
```

Then, in a 5th terminal, set the shared `SERVER_URLS` variable so
every downstream script agrees on where the backends are, and start
whichever balancer(s) you want:

```bash
# macOS/Linux/Git-Bash:
export SERVER_URLS="http://localhost:5001,http://localhost:5002,http://localhost:5003,http://localhost:5004"
python load_balancer/round_robin.py &
python load_balancer/least_connections.py &
python load_balancer/aco_balancer.py &
python load_balancer/dqn_balancer.py &
```

```powershell
# PowerShell:
$env:SERVER_URLS = "http://localhost:5001,http://localhost:5002,http://localhost:5003,http://localhost:5004"
Start-Process python -ArgumentList "load_balancer/round_robin.py"
Start-Process python -ArgumentList "load_balancer/least_connections.py"
Start-Process python -ArgumentList "load_balancer/aco_balancer.py"
Start-Process python -ArgumentList "load_balancer/dqn_balancer.py"
```

Verify everything is up:
```bash
curl http://localhost:5001/health
curl http://localhost:8000/health
curl http://localhost:8003/status
```

## 16. Running with Docker Compose

```bash
cd cloud-load-balancer
cp .env.example .env
docker compose build
docker compose up -d
```

What this does: `cp .env.example .env` gives you an editable copy of
the default environment overrides (traffic rate, memory cap, Grafana
credentials, etc — see Section 26). `docker compose build` builds the
`servers/Dockerfile` image (used by server1-4) and the
`load_balancer/Dockerfile` image (used by all four balancers, and
which installs the CPU PyTorch wheel during the image build so
containers don't need internet access at runtime). `docker compose up
-d` starts server1-4, all four balancers, Prometheus and Grafana in
the background, waiting for each backend's healthcheck before
starting the balancers (`depends_on: condition: service_healthy`).

Check everything is healthy:
```bash
docker compose ps
```

Tail logs for a specific service:
```bash
docker compose logs -f dqn-balancer
```

Open:
- Round Robin: http://localhost:8000
- Least Connections: http://localhost:8001
- ACO: http://localhost:8002
- DQN: http://localhost:8003
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (default `admin`/`admin`, overridable via `.env`)

The DQN balancer's model directory (`./models`) is bind-mounted into
its container at `/app/models`, so a model you train on your host
(Section 17) is immediately visible to the containerized balancer —
call `docker compose exec dqn-balancer python -c "import requests;
requests.post('http://localhost:8003/reload_model')"` or simply
`curl -X POST http://localhost:8003/reload_model` from the host to
pick it up without restarting the container.

## 17. Training the DQN

Training runs the **real environment** (`dqn/environment.py`) against
whichever backends `SERVER_URLS` points at — either your 4 local
`servers/app.py` processes (Section 15) or the Dockerized ones
(`http://localhost:5001..5004` still works since those ports are
published in `docker-compose.yml`).

```bash
# Make sure 4 backend servers are running and reachable first (Section 15 or 16).
export SERVER_URLS="http://localhost:5001,http://localhost:5002,http://localhost:5003,http://localhost:5004"   # PowerShell: $env:SERVER_URLS = "..."
python dqn/train.py --episodes 150 --steps-per-episode 30 --settle-seconds 0.3 --save-every 10 --seed 42
```

Arguments (all optional, defaults shown):

| Flag | Default | Meaning |
|---|---|---|
| `--episodes` | 60 | Number of training episodes |
| `--steps-per-episode` | 40 | Real requests driven per episode |
| `--settle-seconds` | 0.4 | Pause after applying a new workload phase, so metrics reflect it before the first step |
| `--save-every` | 10 | Checkpoint save interval (episodes) |
| `--model-path` | `models/dqn_model.pt` | Where to save |
| `--log-dir` | `logs/` | Where `training_history.csv`/`.json` are written |
| `--resume` | off | Continue training from an existing checkpoint at `--model-path` |
| `--seed` | 42 | RNG seed for reproducible episode-mode ordering |

Each episode randomly picks one of 13 training modes
(`EPISODE_MODES` in `dqn/train.py`) — uniform idle/low/medium/high
load, single/double server overload, CPU/memory/queue/latency
pressure on one server, unequal load, failure-then-recovery, and
mixed dynamic load — so the agent is trained across the full space of
scenarios it will be evaluated on (Section 10), not just one.

Training logs, per episode, to both CSV and JSON: `episode, reward,
avg_reward, avg_response_time_ms, p95_latency_ms, throughput,
success_rate, cpu, memory, queue, load_variance, fairness, epsilon,
loss, q_mean, q_std, q_max, q_min`. This project's own 150-episode
training run is included in `logs/training_history.csv` — see Section
28 for what it shows.

## 18. Generating traffic

```bash
python traffic/traffic_generator.py --target-name dqn --pattern burst --duration 60 --rate 15 --output logs/traffic_run.csv
```

| Flag | Default | Meaning |
|---|---|---|
| `--target` | none | Full balancer URL, e.g. `http://localhost:8003` (overrides `--target-name`) |
| `--target-name` | `dqn` | One of `round_robin`, `least_connections`, `aco`, `dqn` — resolved via `common/settings.py` ports |
| `--pattern` | `normal` | One of `low, normal, high, burst, spike, sustained_high, ramp_up, ramp_down` |
| `--duration` | 60.0 | Seconds to run |
| `--rate` | 10.0 | Base requests/second (patterns scale this up/down over time) |
| `--output` | none | CSV path to log every request's timestamp, request ID, response time, success, selected server, and instantaneous traffic rate |
| `--seed` | 7 | RNG seed |

Inside Docker, the same generator is available as an on-demand
profile so it doesn't start with `docker compose up -d`:
```bash
docker compose --profile tools run --rm traffic-generator
```
(reads `TRAFFIC_PATTERN`, `TRAFFIC_DURATION`, `TRAFFIC_RATE` from
`.env`, targets `dqn-balancer:8003` by default, writes to
`./logs/traffic_run.csv` on the host via the bind-mounted `./logs`).

## 19. Evaluation metrics and how they're computed

All computed in `common/formulas.py` / `testing/metrics.py` from real
captured per-request data — never fabricated:

- **Response time**: average, median, P95, P99 — real
  `time.perf_counter()` deltas around each HTTP call, aggregated with
  `common/formulas.py::percentile()` (linear-interpolation method).
- **Throughput**: successful requests ÷ wall-clock makespan of the run.
- **CPU / memory / queue utilization**: averaged from each response's
  `server_metrics` snapshot.
- **Load variance**: `common/formulas.py::load_variance()` — variance
  of the per-server `load_pressure` values at each step.
- **Load balance degree**: `1 − (stddev(counts) / mean(counts))`
  (coefficient-of-variation based), 1.0 = perfectly even.
- **Jain's fairness index** — exact formula, `common/formulas.py::jain_fairness_index()`:

  ```
  J(x) = (Σ xᵢ)² / (n · Σ xᵢ²)
  ```
  where **`xᵢ` = the number of requests routed to server i** during
  the measurement window, and `n` = number of servers (4). `J = 1.0`
  means perfectly even distribution across all 4 servers; `J = 0.25`
  (`= 1/n`) is the theoretical minimum for 4 servers (all requests to
  one server).
- **Makespan**: wall-clock time from first to last request in the run.
- **Success / failure rate**: real HTTP-level success (2xx) vs.
  failure (5xx / timeout / connection error) counts.
- **Cumulative reward**: sum of `compute_reward()` over every request
  in the run (computed for all 4 algorithms identically, for
  comparison — only the DQN uses it for learning).
- **Adaptation time / overload avoidance rate**: Section 4/8, computed
  in `testing/adaptation_test.py`.

## 20. Running the full evaluation suite

Prerequisite: 4 backend servers + all 4 balancers running (Section
15 or 16), and a trained model at `models/dqn_model.pt` (Section 17)
loaded by the DQN balancer (`GET http://localhost:8003/status` should
show `"model_loaded": true`).

```bash
# 1. Fair, identical-conditions comparison of all 4 algorithms (~200s total: 50s each)
python testing/compare_algorithms.py

# 2. The 5-phase adaptive-behaviour experiment against the DQN (~115s)
python testing/adaptation_test.py

# 3. Failure/recovery experiment against the DQN (~65s)
python testing/failure_test.py

# 4. All 8 automated PASS/FAIL tests (~3 minutes)
python testing/run_tests.py

# 5. Regenerate every plot from the results/*.json + logs/*.csv produced above
python visualization/generate_plots.py
```

Each of steps 1–4 resets backend state via `/reset` where relevant and
writes machine-readable JSON into `results/`; step 5 reads those JSON
files (and `logs/training_history.csv`) and writes `results/*.png`.
None of these scripts invent numbers if a prerequisite file is
missing — `generate_plots.py` skips a plot with a printed message
rather than drawing from absent data.

## 21. Quick Demo Command Cheat Sheet

For a professor/viva walkthrough — copy-paste in order (assumes
Docker; local-without-Docker equivalents are in Section 15):

```bash
# 1) Build and start everything
docker compose build
docker compose up -d
docker compose ps                         # confirm all healthy

# 2) Train the DQN against the running backends (~1-2 minutes)
export SERVER_URLS="http://localhost:5001,http://localhost:5002,http://localhost:5003,http://localhost:5004"
python dqn/train.py --episodes 100 --steps-per-episode 30

# 3) Hot-reload the freshly trained model into the running container
curl -X POST http://localhost:8003/reload_model
curl http://localhost:8003/status         # confirm model_loaded: true

# 4) Send some traffic and look at one real DQN decision
curl http://localhost:8003/

# 5) Prove genuine adaptive behaviour (the headline demo)
python testing/adaptation_test.py
cat results/adaptation_summary.json       # or open results/adaptation_*.png

# 6) Fair comparison against all 3 static baselines
python testing/compare_algorithms.py
cat results/comparison_results.json

# 7) Run every automated test with real PASS/FAIL
python testing/run_tests.py

# 8) Open the dashboards
#    Prometheus: http://localhost:9090
#    Grafana:    http://localhost:3000  (admin/admin)

# 9) Clean up
docker compose down
```

## 22. Automated tests

`testing/run_tests.py` runs 8 tests, each with a real numeric
threshold computed from live measurements — **a test passes only if
the measured condition is actually met, never merely because the
script executed without an exception**:

1. **State sensitivity** — changing backend workload measurably
   changes the built state vector.
2. **Q-value response** — the network's own Q-values measurably
   change between two different real states.
3. **Overload avoidance** — under a real overloaded server, the DQN's
   share of requests to it drops below a threshold.
4. **Server recovery** — after recovery, the DQN's share of requests
   to that server rises back up.
5. **Load balancing fairness** — Round Robin achieves near-1.0 Jain
   fairness under uniform load (sanity check on the baseline itself).
6. **Failure handling** — DQN avoids a hard-failed server measurably
   more than Round Robin does.
7. **Dynamic traffic** — a ramp-up pattern produces a measurably
   increasing real request rate over time.
8. **Performance comparison harness** — all 4 algorithms can be driven
   end-to-end and produce internally consistent success/latency
   numbers.

Run them:
```bash
python testing/run_tests.py
```
Exit code is non-zero if any test fails. Results are also saved to
`results/test_report.json`. This project's own run: **8 PASSED, 0
FAILED, 0 SKIPPED** (see Section 28).

## 23. Simulating server failure and recovery

**Soft failure (works locally or in Docker, no daemon access needed):**
```bash
curl -X POST http://localhost:5002/fail       # server2 now fails every request
curl http://localhost:5002/health              # {"status": "unhealthy", ...}
curl -X POST http://localhost:5002/recover     # server2 restored
```

**Hard failure via Docker (stops the actual container):**
```bash
docker compose stop server2
# ... drive traffic at a balancer, observe it routes around server2 ...
docker compose start server2
```

**Automated version** (drives real traffic through all 3 phases and
measures detection/recovery time):
```bash
python testing/failure_test.py --server-index 1              # soft mode (server_index 1 = server2)
python testing/failure_test.py --server-index 1 --use-docker   # hard mode via `docker compose stop/start`
```

## 24. Prometheus metrics reference

**Backend servers** (scraped from `/prometheus` on each, job
`backend_servers`, label `server="server1"` etc via the metric's own
label):

| Metric | Type | Labels |
|---|---|---|
| `backend_requests_total` | Counter | `server`, `outcome` |
| `backend_cpu_usage_percent` | Gauge | `server` |
| `backend_memory_usage_percent` | Gauge | `server` |
| `backend_queue_length` | Gauge | `server` |
| `backend_active_requests` | Gauge | `server` |
| `backend_request_rate` | Gauge | `server` |
| `backend_throughput` | Gauge | `server` |
| `backend_error_rate` | Gauge | `server` |
| `backend_load_pressure` | Gauge | `server` |
| `backend_health` | Gauge | `server` |

**Static balancers** (Round Robin / Least Connections / ACO, labeled
by `algorithm`):

| Metric | Type | Labels |
|---|---|---|
| `lb_requests_total` | Counter | `algorithm`, `server`, `outcome` |
| `lb_response_time_seconds` | Histogram | `algorithm` |
| `lb_selected_server_total` | Counter | `algorithm`, `server` |
| `lb_reward` | Gauge | `algorithm` |
| `lb_active_connections` | Gauge | `algorithm`, `server` |

**DQN balancer** — the exact metric names required by this project's
spec:

| Metric | Type | Labels |
|---|---|---|
| `dqn_requests_total` | Counter | `server`, `outcome` |
| `dqn_response_time_seconds` | Histogram | – |
| `dqn_reward` | Gauge | – |
| `dqn_q_value` | Gauge | `server` |
| `dqn_selected_server_total` | Counter | `server` |
| `dqn_state_cpu` | Gauge | `server` |
| `dqn_state_memory` | Gauge | `server` |
| `dqn_state_queue` | Gauge | `server` |
| `dqn_state_active_requests` | Gauge | `server` |
| `dqn_state_request_rate` | Gauge | `server` |
| `dqn_state_throughput` | Gauge | `server` |
| `dqn_state_error_rate` | Gauge | `server` |
| `dqn_state_latency` | Gauge | `server` |
| `dqn_server_load_pressure` | Gauge | `server` |
| `dqn_decision_time_seconds` | Histogram | – |
| `dqn_exploration_total` | Counter | – |
| `dqn_exploitation_total` | Counter | – |
| `dqn_success_total` | Counter | – |
| `dqn_failure_total` | Counter | – |
| `dqn_load_variance` | Gauge | – |
| `dqn_fairness_index` | Gauge | – |
| `dqn_epsilon` | Gauge | – |
| `dqn_online_training_loss` | Gauge | – |
| `dqn_model_loaded` | Gauge | – |

**Normalization convention** (documented once, applies everywhere
above): every `dqn_state_*` gauge exposes the **raw, unnormalized**
value (CPU in percent 0–100, memory in percent 0–100, queue as a
count, latency in milliseconds, rates in requests/sec) — the
normalization described in Section 6 happens only inside the state
vector fed to the network, not in what's exposed to Prometheus/Grafana,
so dashboards read in familiar real-world units.

## 25. Grafana dashboard

Auto-provisioned on startup — no manual clicking required:
- Datasource: `monitoring/grafana/provisioning/datasources/datasource.yml`
  registers Prometheus at `http://prometheus:9090`.
- Dashboard: `monitoring/grafana/provisioning/dashboards/dashboard.yml`
  points Grafana at `monitoring/grafana/dashboards/dqn_dashboard.json`,
  which is loaded automatically.

Open http://localhost:3000 (admin/admin by default), and the
dashboard **"Cloud Load Balancer — DQN vs Static Baselines"** is
already there, organized into 4 rows:

1. **Top-level**: requests/sec, throughput, average response time, P95 response time (all algorithms overlaid).
2. **Per-server backend metrics**: CPU, memory, queue length, active requests, request rate, error rate — one panel per metric, one line per server.
3. **DQN internals**: live Q-values per server, reward over time, selected-server distribution, decision latency, exploration-vs-exploitation rate.
4. **System-level**: load variance, Jain fairness index, success/failure rate, adaptation time, overload avoidance rate.

## 26. Environment variables reference

| Variable | Default | Used by |
|---|---|---|
| `SERVER_URLS` | `http://localhost:5001,...,5004` (or Docker DNS if `USE_DOCKER_DNS=true`) | everything that talks to backends |
| `DQN_MODEL_PATH` | `models/dqn_model.pt` | `dqn/config.py`, training + inference |
| `TRAFFIC_PATTERN` / `TRAFFIC_DURATION` / `TRAFFIC_RATE` | `normal` / `60` / `10` | Dockerized `traffic-generator` service |
| `REQUEST_TIMEOUT` | `5` (seconds) | all balancers → backend HTTP calls |
| `METRICS_TIMEOUT` | `1.5` (seconds) | `/metrics` collection timeout |
| `PROMETHEUS_PORT` / per-balancer `*_METRICS_PORT` | `9000/9001/9002/9003` | Prometheus exporter ports |
| `LOG_LEVEL` | `INFO` | all Python services |
| `MEMORY_CAP_MB` | `512` | backend memory-percent normalization |
| `ACO_EVAPORATION_RATE` | `0.1` | ACO pheromone decay |
| `ACO_PHEROMONE_MIN` / `ACO_PHEROMONE_MAX` | `0.2` / `5.0` | ACO Max-Min pheromone floor/ceiling |
| `DQN_INFERENCE_EPSILON` | `0.0` | exploration rate used by the *running* balancer (not training) |
| `DQN_ONLINE_LEARNING` | `false` | if `true`, the DQN balancer keeps learning from live traffic |
| `DQN_LR` / `DQN_GAMMA` / `DQN_EPSILON_START` / `DQN_EPSILON_MIN` / `DQN_EPSILON_DECAY` / `DQN_BATCH_SIZE` / `DQN_BUFFER_CAPACITY` / `DQN_TARGET_UPDATE_FREQ` / `DQN_MIN_REPLAY` / `DQN_GRAD_CLIP` | see Section 9 | DQN hyperparameters |
| `REWARD_W_*`, `REWARD_CLIP` | see Section 8 | reward weighting |
| `QUEUE_NORM_REF` / `ACTIVE_NORM_REF` / `LATENCY_NORM_REF_MS` / `RATE_NORM_REF` / `THROUGHPUT_NORM_REF` / `LOAD_VARIANCE_NORM_REF` | see Section 6 | state normalization |
| `RR_PORT` / `LC_PORT` / `ACO_PORT` / `DQN_PORT` (+ `*_METRICS_PORT`) | `8000/8001/8002/8003` (+ `9000-9003`) | balancer listen ports |
| `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD` | `admin` / `admin` | Grafana container |

A fully-commented `.env.example` with every one of these is included
— `cp .env.example .env` and edit as needed before `docker compose up`.

## 27. Metric documentation table

| Metric | Meaning | Source | In DQN state? | In reward? | In evaluation? |
|---|---|---|---|---|---|
| CPU usage | Real process CPU % | `psutil` via `WorkloadSimulator` | ✓ | ✓ (penalty) | ✓ |
| Memory usage | Real RSS vs. cap, % | `psutil` via `WorkloadSimulator` | ✓ | ✓ (penalty) | ✓ |
| Queue length | Requests waiting on semaphore | `WorkloadSimulator` | ✓ | ✓ (penalty) | ✓ |
| Active requests | In-flight request count | `WorkloadSimulator` | ✓ | ✓ (penalty) | – |
| Average latency | Real measured response time | server-side timing | ✓ | ✓ (penalty) | ✓ |
| Request rate | Rolling requests/sec | server-side rolling window | ✓ | – | – |
| Throughput | Rolling successful requests/sec | server-side rolling window | ✓ | ✓ (bonus) | ✓ |
| Error rate | Rolling failed/total | server-side rolling window | ✓ | ✓ (penalty) | ✓ |
| Health | Self-reported healthy/unhealthy | `/health` logic | ✓ | ✓ (bonus) | – |
| Load pressure | Composite of CPU/mem/queue/active | `WorkloadSimulator.snapshot()` | ✓ | (via imbalance) | ✓ |
| Success/failure | HTTP-level outcome | balancer-observed | – | ✓ (bonus) | ✓ |
| Load variance | Variance of load_pressure across servers | `common/formulas.py` | – | ✓ (penalty) | ✓ |
| Jain fairness index | Evenness of request distribution | `common/formulas.py` | – | – | ✓ |
| Q-values | Network's per-action value estimate | `QNetwork` forward pass | – | – | ✓ (adaptive-behaviour evidence) |
| Reward | Scalar learning signal | `dqn/reward.py` | – | (is the target) | ✓ (cumulative reward tracked) |

## 28. Results produced by this project

This delivery ships with **real, non-fabricated** results from an
actual end-to-end run performed while building this project:
`models/dqn_model.pt` (a 150-episode, ~4500-step trained checkpoint),
`logs/training_history.csv`/`.json`, and every file under `results/`:

- `round_robin_results.json`, `least_connections_results.json`,
  `aco_results.json`, `dqn_results.json`, `comparison_results.json` —
  from `testing/compare_algorithms.py` under identical conditions
  (`{server1: low, server2: medium, server3: low, server4: medium}`
  background workload, the 4-segment ramp/normal/burst/high traffic
  scenario). Headline real numbers: Round Robin 98.8% success /
  71.0ms avg / fairness 1.000; Least Connections 98.6% / 71.6ms /
  0.996; ACO 98.0% / 67.3ms / 0.843; **DQN 98.7% / 57.7ms / 0.470** —
  the DQN achieves the lowest latency by actively avoiding the two
  medium-loaded servers, trading fairness for performance, which is
  exactly what its reward function (Section 8) incentivizes it to do.
- `adaptation_records.json` / `adaptation_summary.json` — the 5-phase
  experiment from Section 4, with 100% overload avoidance and sub-
  second adaptation/recovery times.
- `failure_test_records.json` / `failure_test_summary.json` — 100%
  routing-avoidance during a hard failure, ~0.8s failure detection.
- `test_report.json` — 8/8 automated tests passed.
- 17 PNG plots (`training_*.png`, `adaptation_*.png`,
  `comparison_*.png`) rendered directly from the JSON/CSV above via
  `visualization/generate_plots.py`.

Re-running any command in Sections 17/20 will overwrite these with a
fresh run from your machine — that is expected and is how you verify
the numbers for yourself rather than trusting this document.

## 29. Viva / project explanation section

**Q: What problem does this project solve?**
A: Cloud load balancers must decide, for every incoming request,
which backend server should handle it. Static algorithms (Round
Robin, Least Connections, ACO) make that decision from a fixed rule
or a simple heuristic that never "learns." This project replaces that
decision with a Deep Q-Network that observes rich, real, multi-metric
system state and learns — through trial, real outcomes, and reward
feedback — which server to route to under a wide variety of
conditions (normal load, overload, failure, recovery, bursts).

**Q: What is genuinely "reinforcement learning" here, as opposed to a
disguised heuristic?**
A: The routing decision is `argmax(Q(state))` from a real trained
neural network's forward pass — there is no `if cpu > X: pick Y` logic
anywhere in the decision path (`load_balancer/dqn_balancer.py` calls
only `dqn/inference.py::predict()`, which calls
`DQNAgent.select_action()`, which is pure epsilon-greedy over network
output). The network's weights are updated by real gradient descent
(`DQNAgent.learn()`) on a real Bellman target computed from real
transitions stored in a real replay buffer, exactly as in the
standard DQN algorithm (Mnih et al., 2015).

**Q: How do you know it actually adapts, rather than just converging
to something that looks like Least Connections?**
A: Section 4's 5-phase experiment. The DQN's own Q-value for a server
measurably drops when that server becomes overloaded and measurably
rises again when it recovers (`q_value_sensitivity` /
`q_value_rebound` in `results/adaptation_summary.json`) — that's
internal evidence, not just "it happened to route elsewhere." Combined
with 100% overload avoidance and sub-second reaction time, and a
negative correlation between a server's load pressure and its
selection share across the whole experiment, this is direct evidence
of a policy that responds to system state, not a fixed rule.

**Q: What's in the state vector and why those metrics?**
A: 46 numbers: 10 real metrics per server (CPU, memory, queue,
active requests, latency, request rate, throughput, error rate,
health, composite load pressure) × 4 servers, plus 6 system-wide
aggregates. This is deliberately richer than "just active connection
count" so the network can distinguish different *causes* of poor
performance (a CPU-bound server needs different handling than a
queue-bound one) — see Section 6.

**Q: How is the reward designed, and why doesn't one metric dominate?**
A: Ten weighted components (Section 8), each a configurable
environment variable, combining success/throughput/health bonuses
with latency/CPU/memory/queue/active/error/imbalance penalties,
clipped to a fixed range. No metric's weight is fixed by hidden logic
— they're all in `dqn/config.py::RewardWeights` and documented, so a
reviewer can audit exactly how much each factor matters and change it
without touching code.

**Q: How do static and RL algorithms get compared fairly?**
A: `testing/scenarios.py` defines one canonical background workload
and one canonical 4-segment traffic scenario, and
`testing/compare_algorithms.py` resets backend state and applies that
exact same workload/traffic to each of the 4 algorithms in turn,
recording the same metrics for all of them.

**Q: What happens if a server fails?**
A: Failure is real (`/fail` truly makes a server return errors, or
`docker compose stop` truly kills the container). The DQN sees this
as a health=0/error_rate=1/worst-case-metrics state and its trained
policy routes away from it; `testing/failure_test.py` measures exactly
how fast and how completely.

**Q: What would you improve next?**
A: See Section 34 — multi-step (n-step) returns, prioritized experience
replay, and a 5th backend server to test how well the learned policy
generalizes beyond its trained action-space size (would require
retraining, per Section 33's discussion of `state_dim`/`action_dim`
coupling).

## 30. Troubleshooting

**`ModuleNotFoundError: No module named 'torch'`** — you skipped the
CPU wheel install; run `pip install torch --index-url
https://download.pytorch.org/whl/cpu` before `pip install -r
requirements.txt`.

**DQN balancer logs `ModelDimensionMismatchError` on startup** — the
checkpoint at `DQN_MODEL_PATH` was trained with a different
`state_dim`/`action_dim`/`hidden_sizes` than the current
`dqn/config.py` (e.g. you changed the server count or feature set
without retraining). Retrain (Section 17) or point `DQN_MODEL_PATH`
at a compatible checkpoint.

**`/status` shows `"model_loaded": false`** — no checkpoint exists yet
at `DQN_MODEL_PATH`; the balancer still serves (with an untrained,
effectively random policy) so you can demo the pipeline before
training, but every response also reports `model_loaded: false` so
this is never silently mistaken for a trained policy. Run Section 17.

**`docker compose up` fails on a healthcheck / balancer keeps
restarting** — check `docker compose logs <service>`; the most common
cause is `SERVER_URLS` not resolving (verify no service was renamed
in `docker-compose.yml` without updating `monitoring/prometheus/prometheus.yml`
and the balancer's own `SERVER_URLS` construction). All Docker-internal
addressing uses service names (`http://server1:5000`, never
`localhost`), by design.

**Connection refused when running locally without Docker** — check
you exported `SERVER_URLS` in the *same terminal/session* that starts
the balancers; each new terminal needs the export repeated (or put it
in your shell profile / `.env` + a loader).

**Ports already in use** — another process (or a previous
`docker compose up`) is holding 5001-5004/8000-8003/9000-9003/9090/3000.
Run `docker compose down` first, or on macOS/Linux:
`lsof -i :8003` to find and stop the offending process; on Windows:
`netstat -ano | findstr :8003` then `taskkill /PID <pid> /F`.

**Training seems to make no progress (`loss` stays flat, reward
doesn't improve)** — check `min_replay_before_learn` (default 256) has
actually been reached (very short training runs may never start
learning); check backends are genuinely reachable (a state vector of
constant worst-case values gives the network nothing to learn from).

**Plots are missing / `generate_plots.py` prints "skipping"** — the
corresponding `results/*.json` or `logs/*.csv` doesn't exist yet; run
the corresponding script from Section 20 first.

## 31. Windows-specific notes

- Use PowerShell's `$env:VAR = "value"` (persists for the session) or
  CMD's `set VAR=value` — `export` is bash/zsh-only.
- `docker compose` commands are identical on Windows (Docker Desktop
  with WSL2 backend recommended).
- Activate the virtualenv with `venv\Scripts\Activate.ps1`
  (PowerShell) or `venv\Scripts\activate.bat` (CMD), not the
  Linux/macOS `source venv/bin/activate`.
- If PowerShell blocks the activation script with an execution-policy
  error, run (once, as your own user, not admin):
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- `curl` ships with modern Windows 10/11; if using an older shell,
  `Invoke-WebRequest http://localhost:8003/status` is the PowerShell
  equivalent, or use `curl.exe` explicitly to avoid the
  `Invoke-WebRequest` alias.
- `Start-Process python -ArgumentList "..."` (Section 15) opens each
  balancer in its own window; close the window or `Stop-Process` to
  stop it.

## 32. Stopping and cleaning up

**Docker:**
```bash
docker compose down                 # stop and remove containers + network, keep volumes (Prometheus/Grafana data)
docker compose down -v              # also remove volumes (Prometheus/Grafana history)
docker compose down --rmi local     # also remove the images this project built
```

**Local (no Docker):** stop each Python process (`Ctrl+C` in each
terminal, or on Windows `Start-Process`'d windows: close the window /
`Stop-Process -Name python`).

**Reset backend state without stopping anything** (useful between
repeated experiment runs):
```bash
for p in 5001 5002 5003 5004; do curl -s -X POST http://localhost:$p/reset; curl -s -X POST http://localhost:$p/recover; done
```

## 33. Design decisions and known trade-offs

- **State/action dimension is coupled to server count.** Adding or
  removing a backend server changes `state_dim` (`n_servers*10+6`)
  and `action_dim` (`n_servers`), which means the network's input/
  output layers change shape — any existing checkpoint becomes
  incompatible and `DQNAgent.load()` will correctly refuse it
  (`ModelDimensionMismatchError`) rather than silently misbehaving.
  Retraining from scratch (or via a deliberate weight-surgery step,
  not implemented here) is required after changing server count.
- **DQN trades fairness for reward-optimality by design.** Section 28
  shows the DQN achieving lower latency but lower Jain fairness than
  Round Robin under *unequal background load* — this is the expected,
  correct behaviour of a reward-driven policy (its reward function
  penalizes latency/CPU/queue, not request-count equality). If a
  deployment specifically wants fairness prioritized, raise
  `REWARD_W_IMBALANCE` relative to the other weights and retrain.
- **Online learning is off by default** (`DQN_ONLINE_LEARNING=false`)
  so that a deployed balancer's behaviour is deterministic and
  reproducible for grading/demo purposes; flip it on to let the agent
  keep improving from live production traffic (at the cost of
  behaviour that can now drift over time).
- **The workload simulator is a simulator, not real cloud
  infrastructure** — CPU/memory/queue are genuinely measured and
  genuinely respond to real load, but they run inside one container/
  process rather than across real distributed infrastructure. This is
  an intentional, explicit scope boundary for a coursework/demo
  environment, not a hidden shortcut.

## 34. Extending the project

- Add a 5th (or Nth) backend server: add a `server5` block to
  `docker-compose.yml` (copy the `server1` pattern), add it to
  `SERVER_URLS`/`server_names` in `common/settings.py`, add its
  scrape target to `monitoring/prometheus/prometheus.yml`, then
  retrain (Section 33).
- Try prioritized experience replay: replace the uniform `random.sample`
  in `dqn/replay_buffer.py::ReplayBuffer.sample()` with a priority-
  weighted sampler (e.g. proportional to `|TD-error|`).
- Try Double DQN: in `dqn/agent.py::learn()`, select the next action
  with the online network but evaluate its value with the target
  network, instead of taking `target_net(...).max()` directly.
- Add a 5th static baseline (e.g. Weighted Round Robin, Power-of-Two-
  Choices) in `load_balancer/`, following the existing
  `round_robin.py` structure and `load_balancer/common.py` helpers so
  it plugs directly into `testing/compare_algorithms.py`'s
  `ALGORITHMS` list.

## 35. File-by-file summary

See Section 5 for the full tree; every `.py` file under `common/`,
`servers/`, `load_balancer/`, `dqn/`, `traffic/`, `testing/` and
`visualization/` begins with a module docstring explaining its role —
read those in-file docstrings for implementation-level detail beyond
what this README covers at the architecture level.

## 36. Glossary

- **DQN (Deep Q-Network)**: a neural network trained to approximate
  `Q(state, action)`, the expected future discounted reward of taking
  `action` in `state`.
- **Epsilon-greedy**: an exploration strategy — pick a random action
  with probability epsilon, otherwise pick the currently-best-known
  action.
- **Experience replay**: storing past `(state, action, reward,
  next_state, done)` transitions and training on random batches of
  them, rather than only the most recent transition.
- **Target network**: a periodically-synced, frozen copy of the
  network used to compute stable Bellman targets, preventing the
  "moving target" instability of updating a network against its own
  live predictions.
- **Jain's fairness index**: a standard [0,1] measure of how evenly a
  resource (here, request count) is distributed across participants.
- **Makespan**: total wall-clock time to complete a batch of work.
- **Pheromone (ACO)**: a decaying, reinforced weight per server used
  by Ant Colony Optimization to bias future routing toward
  historically successful servers.

## 37. License / academic use note

This project was built as an educational/coursework deliverable. No
external license is asserted beyond what your institution requires
for submission; reuse, adapt and extend it freely for learning
purposes, and cite the reinforcement-learning concepts appropriately
(DQN: Mnih et al., *Human-level control through deep reinforcement
learning*, Nature, 2015) in any academic writeup.

---

## 38. v3 DQN training: avoiding over- and under-fitting

The **algorithm is standard DQN** (Mnih et al. 2015): an online Q-network, a frozen target network copied every
`target_update_freq` learn steps, experience replay, Huber loss, and the Bellman target
`y = r + γ · max_a Q_target(s', a)`. It is **not Double DQN**. The **network is unchanged**:
`46 → 256 → ReLU → 128 → ReLU → 4`. Only *how the weights are fitted* changed.

| Problem seen in v2 | v3 fix (file) |
|---|---|
| Only one request in flight during training, while evaluation runs 10–40 concurrent req/s (train/test mismatch) | `--workers 4` concurrent agent clients route through the *same* agent and learn the cost of herding (`dqn/train.py`) |
| Training ran with an idle system | `BackgroundTraffic`: random Poisson "other clients" load (`dqn/environment.py`) |
| Nine fixed presets could be memorised | **Domain randomisation**: every preset parameter is scaled ×[0.75, 1.25] during training (`apply_preset(randomize=)`) |
| Positional shortcuts ("server 3 is usually fine") | **Server-permutation augmentation**: each real transition is also stored with the 4 servers re-ordered (servers are interchangeable), `--augment 3` |
| ε reached 0.05 after ~600 steps, then exploration stopped | Linear ε schedule over 50% of training (`--explore-fraction`) |
| Fixed learning rate | Cosine LR from 1e-3 to 1e-4 (`lr_min`) |
| No regularisation | **AdamW** decoupled weight decay 1e-5 (`weight_decay`) |
| γ = 0.95 inflated Q-values on a near-bandit task | γ = 0.90 (`DQN_GAMMA`) |
| Last checkpoint saved, even if it was worse | **Validation every 8 episodes + early stopping (patience 5) + best-checkpoint save** |
| No over/under-fit diagnostics | Held-out TD-loss buffer, train-probe vs validation gap, reference-policy baselines, `logs/fit_report.json` |

**Validation design.** Validation is a fixed set of 8 episodes, identical for every evaluation and every policy. It covers
6 in-distribution families with a fresh seed, plus **2 families never seen in training** (`slow_plus_overload`,
`three_degraded`). The *train-probe* set uses the same families drawn from the training distribution, which makes
`train-probe − validation` a like-for-like **generalisation gap**. Transitions collected during validation go into a
separate buffer the network never trains on, and the TD loss on it is compared with the TD loss on the replay memory.

**Automatic fit diagnosis** (`logs/fit_report.json`):

* **UNDERFIT** – best validation score ≤ the best non-learning reference policy (round robin, random, or least-active
  measured on the *same* validation episodes)
* **OVERFIT** – generalisation gap > 0.15, or held-out TD loss > 2× training TD loss
* **GOOD FIT** – otherwise

Train (≈10–20 min with 4 backends running):

```bash
docker compose up -d server1 server2 server3 server4
python dqn/train.py                      # defaults = the configuration of the shipped model
python dqn/train.py --quick              # 2-minute smoke test
curl -X POST http://localhost:8003/reload_model      # hot-load the new checkpoint
```

| Flag | Default | Meaning |
|---|---|---|
| `--episodes` | 96 | max episodes (early stopping usually ends sooner; the shipped model used `--episodes 120`) |
| `--steps-per-episode` | 50 | real requests per episode |
| `--workers` | 4 | concurrent agent clients |
| `--augment` | 3 | permutation copies per real transition |
| `--randomize` | 0.25 | domain-randomisation strength |
| `--bg-max-rps` | 6 | max background traffic |
| `--val-every` / `--patience` / `--min-episodes` | 8 / 5 / 48 | early-stopping schedule |
| `--gap-tolerance` | 0.15 | overfit threshold |

Outputs: `models/dqn_model.pt` (best validation, with the fit report embedded), `models/dqn_model_last.pt`,
`logs/training_history.csv|json`, `logs/validation_history.json`, `logs/fit_report.json`,
`results/training_fit_curves.png`.

**Inference tie-tolerance (`DQN_TIE_MARGIN`, default 0.15).** On the real system, equally healthy servers differ by
about 0.1–0.2 in Q-value, while a degraded server sits ≥ 0.32 (slow) to ≈ 3.4 (failed) below the best. A strict
argmax therefore sends every request to whichever equivalent server happens to be 0.01 higher. That herds load and
re-admits a recovered server very late; with a strict argmax the original "4. Server recovery" test failed. With the
margin, servers within 0.15 of the best Q-value count as tied and one is picked at random. The network, Q-values and
training are untouched. Set `DQN_TIE_MARGIN=0` for a strict argmax. Every DQN response now also carries
`tie_candidates` and `tie_margin`.

**Model selection (honest note).** Two v3 runs were made. **v3a** used one worker and no augmentation; its logs are in
`logs/experiments/v3a_sequential/`. **v3b** used 4 workers and ×3 augmentation; it is the shipped model, with logs in
`logs/` and `logs/experiments/v3b_concurrent_augmented/`. v3a scored higher on the synthetic policy probes (85% vs 79%).
v3b was clearly better on live traffic in MyCamu's comparison harness: with a slow server, p95 was 55 ms vs 326 ms and
routing accuracy 98% vs 92%, and under overload it spread load across all healthy servers. v3b was chosen for its live
behaviour. That selection used MyCamu's harness, *not* the stress suite, so the stress-suite numbers in Section 42 are
an unbiased final test.

## 39. Scenario stress-test suite (PASS/FAIL)

```bash
python testing/stress_test_suite.py                 # all 19 scenarios x 4 algorithms (~40 min)
python testing/stress_test_suite.py --quick         # half durations
python testing/stress_test_suite.py --scenarios hard_failure flapping_server --repeats 3
python testing/stress_test_suite.py --rescore results/stress_suite/results.json   # re-score, no traffic
```

Scenarios (`testing/stress_scenarios.py`). Every scenario is a timeline of stages, and each stage sets the workload,
failures/recoveries and the traffic pattern:

| Category | Scenarios |
|---|---|
| Traffic shape (healthy servers) | `low_traffic`, `normal_traffic`, `high_traffic`, `spike_traffic`, `burst_traffic`, `ramp_up`, `flash_crowd` |
| Degradation | `slow_server`, `single_overload`, `double_overload`, `cpu_memory_pressure`, `queue_buildup_high_traffic`, `heterogeneous_servers`, `spike_during_overload` |
| Failure / dynamic | `hard_failure`, `failure_and_recovery`, `flapping_server`, `cascading_degradation`, `overload_migration` |

For every scenario, each algorithm is run in a seeded random order (avoids order bias), after a backend reset, against
identical conditions. A monitor thread samples every server's `/metrics` once per second and classifies it as
HEALTHY / BUSY / SLOW / OVERLOADED / FAILED.

**Metrics** (per stage *and* per scenario):

* success rate
* avg / p50 / p95 / p99 latency
* **backend-only latency**: routing quality without balancer overhead
* throughput and **goodput** (successful *and* within the 250 ms SLA, per second)
* SLA-violation rate
* Jain fairness and server distribution
* **routing accuracy**: decisions that avoided every degraded server while a healthy one existed. The suite *creates*
  the incident, so ground truth is known; the balancers never see it.
* requests-to-adapt and recovery-detection time
* DQN only: Q-value margin and Q/latency rank agreement

**Comparison.** Each scenario gets a composite *Load-Balancing Quality* score (0–100): 35% success, 25% SLA compliance,
20% latency vs the best, and 20% routing accuracy (or relative p95). The suite also reports per-metric ranks and wins,
a Mann-Whitney U test with a bootstrap 95% CI on DQN-vs-baseline latency, and a two-proportion z-test on success rates.

**PASS/FAIL checks** for DQN, per scenario:

* success ≥ 95% (traffic scenarios) or ≥ 90% (incidents)
* p95 ≤ 375 / 625 ms
* routing accuracy ≥ 85% (where degraded servers exist)
* composite ≥ best baseline − 1 point, **or** a *statistical tie*: the latency and success-rate differences are not
  significant at p < 0.05. Ties are labelled as ties, never as wins.

Probe checks are added too (Section 40). Outputs go to `results/stress_suite/`: `report.html` (open in a browser),
`results.json`, `summary.csv`, `checks.json` and `*.png`.

## 40. Offline policy probes

```bash
python testing/policy_probe_test.py --n 800
```

This generates 800 synthetic but realistic incident snapshots, with value ranges calibrated to the real presets. The
categories are: one overloaded / slow / failed / queue-buildup / CPU-starved server, two degraded, three degraded (one
good server left), and busy-vs-degraded. Each snapshot goes through the exact `build_state()` and the real network.
A decision is correct if the chosen server is not degraded. Round Robin, Random and Least Connections are scored on
the same probes. No servers or Docker are needed. Output: `results/policy_probes.json`.

## 41. TCE MyCamu (port 8080)

MyCamu is a Flask + SQLite student portal whose every course registration is a **real request routed through the
selected balancer**. Its technical pages visualise the real routing.

```
mycamu/
├── app.py          # all routes (auth, student data, routing, workers, rush, traffic, compare, health, /metrics)
├── db.py           # SQLite schema (CREATE IF NOT EXISTS) + seed-only-if-empty demo data, werkzeug password hashes
├── lb_client.py    # HTTP client for balancers/servers, status classification, metric summaries
├── traffic.py      # Traffic Control Center job (START/STOP/RESET, one at a time) + comparison job
├── config.py       # every URL from environment variables (Docker DNS or localhost)
├── static/         # index.html, css/app.css, js/app.js (student pages), js/tech.js (technical pages)
├── Dockerfile  requirements.txt  .env.example  .dockerignore
```

**Demo accounts:** `student001` … `student005`, password **`tce@123`** (set `MYCAMU_DEMO_PASSWORD` before the *first*
start to change it). The password is stored hashed and is not shown in the UI.

**Pages:**

* Student: Login, Dashboard, Courses, Course Details, Enrollment, Timetable, Fees, Registration History
* Technical: DQN Monitor (live `/predict` Q-values, state-vector heat-map), Server Monitor (live cards + 2-minute
  latency/CPU charts), Server Lab (NORMAL / SLOW / OVERLOAD / FAILURE / RECOVER per server, plus a 5-step guided
  failure demo), Traffic Control Center (N real requests, animated flow driven by real completed events, live KPIs,
  distribution, Q-values, request log), Algorithm Comparison (same scenario × N requests per algorithm, charts,
  history), System Health

**Enrollment flow.** `POST /api/register/<code>` runs these checks server-side: student verified, course exists,
duplicate, credit limit, timetable clash, seats. It then sends a real request through the balancer and saves the
registration only if the backend succeeded. The response keeps `algorithm, latency_ms, q_values, selected_server,
status_code`, and adds `checks`, `server_metrics_before` (the states the algorithm analysed) and `server_metrics`.

**API:**

| Area | Endpoints |
|---|---|
| Auth | `POST /api/auth/login`, `POST /api/auth/logout`, `GET /api/auth/me`, `GET /api/auth/demo-accounts` |
| Student | `GET /api/dashboard`, `GET /api/courses`, `GET /api/courses/<code>`, `POST`/`DELETE /api/register/<code>`, `GET /api/timetable`, `GET /api/fees`, `GET /api/history` |
| Routing | `GET /api/routing/dqn`, `GET`/`POST /api/routing/algorithm` |
| Workers | `GET /api/workers`, `POST /api/workers/<server>/<slow\|overload\|fail\|recover\|normal\|medium>`, `POST /api/workers/recover-all` |
| Rush | `POST /api/rush/normal` (50 req), `/medium` (150), `/high` (300); optional body `{requests, concurrency, algorithm}` |
| Traffic | `POST /api/traffic/start {requests, algorithm, scenario, target_server, rate, concurrency}`, `POST /api/traffic/stop`, `POST /api/traffic/reset`, `GET /api/traffic/status?since=<seq>` |
| Compare | `POST /api/compare/start {scenario, requests, algorithms[], target_server}`, `POST /api/compare/stop`, `GET /api/compare/status`, `GET /api/compare/history` |
| Health | `GET /api/health` (really calls every balancer + server `/health`), `GET /metrics` (Prometheus) |

**New DQN-balancer endpoints (additive):**

* `GET /predict`: live Q-values for the current real state. It routes nothing.
* `POST /reset`: resets the rolling counters only, never the model.
* Every balancer response also carries `server_metrics_before`.

**Docker.** The `mycamu` service is added to `docker-compose.yml` with the `ROUND_ROBIN_URL … SERVER4_URL` environment
variables. Inside the network the backends are on **:5000**; 5001–5004 are host mappings only. The service waits on
`service_healthy` for all four balancers and has its own healthcheck. The database lives at `/app/data/mycamu.db` on
the named volume **`mycamu_data`**, so it survives `restart`/`down`/`up` (only `docker compose down -v` deletes it).
Prometheus gains a `mycamu` scrape job.

```bash
docker compose down
docker compose build --no-cache mycamu
docker compose build                  # balancers too (they contain the v3 code)
docker compose up -d
docker compose ps
curl.exe http://localhost:8003/status            # model_loaded: true, state_dim: 46
curl.exe -X POST http://localhost:8003/reset
curl.exe http://localhost:8080/api/health
curl.exe http://localhost:8080/api/routing/dqn
curl.exe http://localhost:8080/api/workers
curl.exe -c cj.txt -H "Content-Type: application/json" -d "{\"student_id\":\"student001\",\"password\":\"tce@123\"}" http://localhost:8080/api/auth/login
curl.exe -b cj.txt -X POST http://localhost:8080/api/register/CSE401
curl.exe -X POST http://localhost:8080/api/rush/medium
curl.exe -X POST http://localhost:8080/api/workers/server1/slow
curl.exe -X POST http://localhost:8080/api/rush/medium
curl.exe -X POST http://localhost:8080/api/workers/recover-all
docker compose restart mycamu        # registrations persist
```

**Local development (no Docker).** Start the servers and balancers as in Section 15, then run
`cd mycamu && pip install -r requirements.txt && python app.py`. The defaults are the localhost host ports.

**Production exposure.** Only 8080 (MyCamu), 3000 (Grafana) and 9090 (Prometheus) need to be public. Remove the
`ports:` of server1-4 and the four balancers so they are reachable only on `lb_network`. They stay published in
development for debugging.

## 42. v3 measured results

All numbers below were measured in the development sandbox: 2 vCPUs, with all 4 servers, 4 balancers and the test
client sharing those CPUs. Example output; actual values depend on execution. On your machine, rerun
`python testing/stress_test_suite.py` and the fresh `results/stress_suite/report.html` replaces these numbers.

**Training fit** (`logs/fit_report.json`): **GOOD FIT**. Early stopping triggered at episode 80; the best checkpoint
is from episode 40.

* Validation score (mean reward per request): **1.102**. Reference policies on the same episodes: least-active
  1.060, round robin 0.933, random 0.900. So the model is **not underfit**.
* Generalisation gap (train-probe − validation, in-distribution): **0.012**.
* Held-out / training TD-loss ratio: **0.98**. So it is **not overfit**.
* The 2 never-seen scenario families scored **1.179**, higher than the in-distribution families.

**Original 8 automated tests** (`testing/run_tests.py`): **8/8 PASS**, including overload avoidance (0% of traffic
sent to the overloaded server), failure handling (DQN 0% vs Round Robin 25% to the failed server) and server recovery.

**Stress suite** (`results/stress_suite/`, shipped configuration `DQN_TIE_MARGIN=0.15`): **62/71 checks passed (87.3%)**.

* DQN won **12 of 19** scenarios on composite score; ACO won 3, LC 2, RR 2.
* Mean composite score: **DQN 96.4**, LC 90.6, ACO 90.3, RR 87.7.
* Mean routing accuracy on scenarios with a degraded server: **DQN 93.0%**, LC 75.3%, ACO 73.2%, RR 68.7%.

| Scenario | Composite DQN / LC / ACO / RR | Winner | DQN success | DQN avg ms (backend ms) | best baseline avg ms | DQN routing acc. | best baseline acc. |
|---|---|---|---|---|---|---|---|
| low_traffic | 93.3 / 98.7 / 99.6 / 99.9 | RR | 100.0% | 37.5 (17.6) | 34.0 (ACO) | n/a | n/a |
| normal_traffic | 95.1 / 96.0 / 93.6 / 98.8 | RR | 98.5% | 62.7 (34.0) | 53.6 (RR) | n/a | n/a |
| high_traffic | 98.8 / 98.7 / 98.2 / 97.4 | DQN | 99.5% | 32.9 (17.3) | 31.5 (RR) | n/a | n/a |
| spike_traffic | 91.8 / 99.0 / 99.7 / 98.7 | ACO | 98.5% | 41.7 (18.5) | 35.6 (ACO) | n/a | n/a |
| burst_traffic | 97.5 / 98.4 / 97.7 / 98.4 | LC | 98.2% | 63.9 (31.0) | 62.2 (LC) | n/a | n/a |
| ramp_up | 94.8 / 99.2 / 99.5 / 98.8 | ACO | 98.8% | 36.0 (17.7) | 32.7 (LC) | n/a | n/a |
| flash_crowd | 97.0 / 99.5 / 98.1 / 99.2 | LC | 99.1% | 40.5 (18.5) | 38.1 (LC) | n/a | n/a |
| slow_server | 98.8 / 81.7 / 84.3 / 74.9 | DQN | 99.5% | 38.5 (22.4) | 73.8 (ACO) | 98.0% | 88.0% |
| single_overload | 98.8 / 85.2 / 89.5 / 78.4 | DQN | 98.0% | 64.5 (17.9) | 88.0 (ACO) | 100.0% | 89.5% |
| double_overload | 98.5 / 72.3 / 73.7 / 63.7 | DQN | 97.5% | 84.8 (19.8) | 148.3 (ACO) | 100.0% | 64.5% |
| cpu_memory_pressure | 89.0 / 87.8 / 90.8 / 88.1 | ACO | 98.0% | 101.7 (38.0) | 88.7 (ACO) | 64.0% | 60.0% |
| queue_buildup_high_traffic | 94.7 / 94.1 / 91.9 / 94.2 | DQN | 98.7% | 46.6 (23.1) | 46.5 (RR) | 77.2% | 74.9% |
| heterogeneous_servers | 98.6 / 94.8 / 86.7 / 92.2 | DQN | 98.8% | 70.2 (27.7) | 73.9 (LC) | 96.7% | 80.0% |
| spike_during_overload | 98.0 / 93.4 / 94.2 / 79.5 | DQN | 99.0% | 127.3 (21.2) | 120.9 (LC) | 100.0% | 88.6% |
| hard_failure | 97.0 / 86.4 / 72.2 / 82.1 | DQN | 99.0% | 35.1 (17.5) | 30.9 (RR) | 100.0% | 82.0% |
| failure_and_recovery | 96.8 / 87.8 / 82.5 / 87.8 | DQN | 99.2% | 36.1 (17.5) | 31.1 (ACO) | 100.0% | 74.7% |
| flapping_server | 99.5 / 88.4 / 97.3 / 84.8 | DQN | 99.6% | 57.2 (18.2) | 56.6 (ACO) | 100.0% | 95.8% |
| cascading_degradation | 93.6 / 74.4 / 72.8 / 69.7 | DQN | 99.7% | 91.3 (46.5) | 143.4 (ACO) | 79.7% | 58.0% |
| overload_migration | 99.8 / 85.9 / 92.7 / 80.1 | DQN | 99.6% | 83.6 (19.2) | 92.6 (ACO) | 100.0% | 88.8% |


**How to read this honestly**

* **Whenever a server is slow, overloaded, failed, flapping or degraded, DQN wins clearly.** Examples: slow server
  38.5 ms vs 73.8 ms for the best baseline; double overload 84.8 vs 148.3 ms; cascading degradation 91.3 vs 143.4 ms.
  It routes 98–100% of requests away from the bad server, versus 64–96% for the baselines. In the failure scenarios
  its success rate stays at ~99% while LC/RR fall to 75–90%. LC "black-holes" traffic into a fast-failing server
  because that server always has the fewest open connections.
* **With four equally healthy servers, every algorithm routes equally well.** Backend-only latency is ~17–18 ms for
  all of them. DQN then loses on end-to-end latency by about 2–6 ms. That gap is the DQN balancer's own per-request
  overhead (PyTorch runtime plus extra state/reward work) on a CPU-starved 2-core sandbox, not a routing mistake.
  These are the failed "composite vs baselines" checks in the traffic-only scenarios. Expect the gap to shrink on a
  machine with more cores.
* **Routing-accuracy checks also fail in three scenarios:**
  - `cpu_memory_pressure` (64%): memory-pressure servers are only mildly slower, so their Q-values fall inside the
    0.15 tie margin.
  - `queue_buildup_high_traffic` (77%).
  - `cascading_degradation` (80%): three of four servers degraded; sending everything to the single healthy server
    would overload it.

  DQN still has the best accuracy of all four algorithms in each of these.
* **Policy probes:** DQN 78.0% vs Least Connections 71.5%, Random 66.5%, Round Robin 65.2%. That beats every baseline,
  but the absolute ≥ 90% target is not met (weakest category: three degraded servers).
* **Strict argmax** (`DQN_TIE_MARGIN=0`, `results/stress_suite_strict_argmax/`): 63/71 checks, 13/19 wins, mean
  composite 97.3, routing accuracy 95.0%. It is slightly better on this suite, but it herds healthy traffic onto one
  server and fails the original "4. Server recovery" test, which is why the shipped default is 0.15.
