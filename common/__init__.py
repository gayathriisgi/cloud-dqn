"""Shared, cross-cutting configuration and helpers used by every component
(backend servers, all four load balancers, the DQN package, the traffic
generator and the testing/visualization tools).

Keeping this in one place guarantees that server URLs, ports, timeouts and
normalization constants can never drift apart between the pieces that
build the DQN state, the pieces that compute the reward and the pieces
that scrape Prometheus - which is one of the most common sources of
"looks-like-RL-but-is-actually-broken" bugs in projects like this one.
"""
