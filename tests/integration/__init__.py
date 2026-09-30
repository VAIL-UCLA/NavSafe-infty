"""Integration tests for NexusSim.

These tests exercise full pipelines (IsaacSim runtime, asset catalogs,
end-to-end evaluation, etc.) and are NOT user tools. Files in this
directory are pytest-discoverable: file names start with ``test_`` so
pytest picks them up under ``pytest tests/integration``.

Some integration scripts in this directory are inherently subprocess
or IsaacSim-runtime style (they call ``argparse.parse_args()`` or
boot ``AppLauncher`` at import time). pytest collection for those
files is suppressed via ``conftest.py`` in this directory.
"""
