"""Consolidated, authoritative DQN implementation for the cloud load
balancer project.

This package replaces every earlier, competing DQN implementation that
used to live scattered across ``dqn/`` (NumPy), ``dqn_test/`` (partial
PyTorch experiments) and ``servers/dqn1/`` (a second, incompatible
PyTorch environment/model pair). Those files are kept out of the
shipped project; see ``README.md`` -> "Project structure" for the
mapping from old files to the modules below.

Modules
-------
config.py          Hyperparameters, reward weights, state layout - all
                    in one place, overridable via environment variables.
state.py           Builds and documents the normalized DQN state vector
                    from real backend `/metrics` JSON.
model.py           The Q-network (PyTorch).
replay_buffer.py   Uniform-sampling experience replay buffer.
reward.py          The multi-objective reward function.
agent.py           DQNAgent: epsilon-greedy action selection, learning
                    step (target network + Bellman update), save/load
                    with strict state-dimension metadata checking.
environment.py      Drives real HTTP requests against the real backend
                    Flask servers through a schedule of workload
                    scenarios (low/medium/high/burst/ramp/overload/
                    failure/recovery) used for training and for the
                    adaptive-behaviour experiment.
train.py           CLI training entrypoint.
inference.py        Thin wrapper used by load_balancer/dqn_balancer.py
                    to load a trained model and serve Q-values.
"""
