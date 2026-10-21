"""Standalone VLA inference servers (JSON over stdin/stdout).

Each server script runs under the dedicated VLA venv
(``NAVSAFE_VLA_PYTHON``, default ``~/.cache/navsafe/venvs/vla/bin/python``;
ResWorld uses its own, ``NAVSAFE_RESWORLD_PYTHON``) in a persistent
subprocess, following the Alpamayo-R1 adapter pattern: the host adapter
writes one JSON request per line to stdin and reads one JSON response
per line from stdout.  The scripts are import-standalone — they never
import ``navsafe`` (the venv has no IsaacSim/navsafe deps); vendored
model code is reached through explicit ``sys.path`` entries.

Protocol (all servers): print ``READY`` once loaded; then per line
``{"...request...}`` -> ``{"trajectory": ...}``, or ``{"error": ...,
"traceback": ...}``.  Each script's docstring gives its request fields and
the shape and frame of its trajectory; the matching adapter in
``navsafe/policy/sensor`` converts that to the NavSafe contract.
"""
