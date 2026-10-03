"""Consolidated load-balancer implementations.

Four independent Flask services, all built on the same metric-collection
and request-forwarding helpers (``load_balancer/common.py``) and all
evaluated under identical conditions (same 4 backend servers, same
traffic scenarios, same evaluation duration - see ``testing/``):

* ``round_robin.py``        - static, fixed-order rotation.
* ``least_connections.py``  - static, greedy on live active-connection count.
* ``aco_balancer.py``       - static, pheromone-weighted probabilistic choice.
* ``dqn_balancer.py``       - learned: multi-metric state -> DQN Q-values
  -> action, using ``dqn/inference.py``.

None of the three static algorithms are "converted to RL" - they remain
fixed decision procedures, which is the whole point of using them as a
baseline for the DQN comparison in ``testing/compare_algorithms.py``.
"""
