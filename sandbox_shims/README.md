# sandbox_shims/

Optional. Only needed if you're running this project somewhere with no
network access to PyPI (real `torch`) or a container registry (real
`prom/prometheus` / `grafana/grafana` Docker images) -- e.g. the sandbox
this evaluation was run in.

On a normal machine with internet access, ignore this folder entirely:
`pip install -r requirements.txt` gets you the real PyTorch, and
`docker compose up -d` gets you real Prometheus + Grafana. Nothing in
`dqn/`, `load_balancer/`, `servers/`, or `testing/` was changed to
accommodate these shims -- they only stand in for two third-party
packages via Python's import system (`PYTHONPATH=sandbox_shims`).

- `torch/` -- a numpy-backed reimplementation of the small subset of
  PyTorch this project uses (Linear/ReLU/Sequential, real forward +
  backward autodiff, Adam, and a pickle-based save/load compatible with
  this project's checkpoint format). Verified against the real
  `models/dqn_model.pt` checkpoint: state_dict keys match exactly and
  loaded Q-values vary sensibly across inputs.
- `prometheus_client.py` -- implements the real Prometheus text
  exposition format (including real cumulative histogram buckets) and
  actually binds an HTTP server on `start_http_server()`, so a real
  Prometheus *server* binary (downloaded directly, since only the
  Docker image was blocked) can genuinely scrape these processes.

Usage: prepend it to PYTHONPATH before running any project script, e.g.
`PYTHONPATH=sandbox_shims:. python3 load_balancer/dqn_balancer.py`.
