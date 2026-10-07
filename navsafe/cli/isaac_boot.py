# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Shared IsaacSim boot helpers for NavSafe entry-point scripts.

This module lives **outside** the ``navsafe`` package on purpose. IsaacSim must
be booted before any ``navsafe`` import, because ``import navsafe`` eagerly
pulls numpy + torch (via the policy registry), and loading those before the sim
app boots corrupts USD's Boost.Python type registration ("No to_python converter
for UsdTimeCode"). It also fixes the glib load-order issue (IsaacSim's bundled
``libgobject`` needs ``g_dir_unref`` / glib >= 2.80; older system glib fails GPU
Foundation init) via an ``LD_PRELOAD`` re-exec. Only stdlib + ``isaacsim`` are
imported here, so it is safe to import first.

Entry-point scripts (which run as ``python scripts/<dir>/<script>.py``) reach it
by putting the ``scripts/`` dir on ``sys.path`` before importing::

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from isaac_boot import boot_isaac_if_available          # full boot, or
    from isaac_boot import ensure_isaac_glib_preload        # just the glib fix
"""

from __future__ import annotations

import glob
import importlib.util
import os
import subprocess
import sys

# Sentinel marking that the glib-preload check already ran in this process tree,
# so the os.execv re-exec below cannot loop.
_GLIB_PRELOAD_SENTINEL = "NAVSAFE_ISAAC_GLIB_PRELOADED"
# Symbol present only in glib >= 2.80; IsaacSim's bundled libgobject needs it.
_GLIB_REQUIRED_SYMBOL = "g_dir_unref"


def _find_isaac_bundled_glib() -> list[str]:
    """Return ``[libglib, libgobject]`` from IsaacSim's ``omni.gpu_foundation`` deps.

    These bundled copies export :data:`_GLIB_REQUIRED_SYMBOL` (glib >= 2.80).
    Returns ``[]`` if IsaacSim isn't importable / the libs aren't found.
    """
    spec = importlib.util.find_spec("isaacsim")
    for base in list(getattr(spec, "submodule_search_locations", None) or []):
        deps = os.path.join(base, "extscache", "omni.gpu_foundation-*", "bin", "deps")
        glibs = sorted(glob.glob(os.path.join(deps, "libglib-2.0.so.0")))
        gobjs = sorted(glob.glob(os.path.join(deps, "libgobject-2.0.so.0")))
        if glibs and gobjs:
            return [glibs[-1], gobjs[-1]]
    return []


def ensure_isaac_glib_preload() -> None:
    """Re-exec with the bundled glib on ``LD_PRELOAD`` when the ambient glib is too old.

    ``LD_PRELOAD`` must be set before the process starts, so if a fix is needed we
    set it and ``os.execv`` a fresh interpreter (sentinel-guarded → runs at most
    once). No-op when the ambient glib already has :data:`_GLIB_REQUIRED_SYMBOL`
    or the bundled libs can't be found. Call this before booting ``AppLauncher``.
    """
    if os.environ.get(_GLIB_PRELOAD_SENTINEL):
        return
    # If the ambient glib already provides the symbol, no preload is needed.
    # Probe it in a SUBPROCESS: a ctypes.CDLL in this process leaves the system
    # glib resident, and kit's RTX plugin chain (shaderdb -> materialdb ->
    # scenedb -> raytracing) then binds against it instead of IsaacSim's
    # bundled glib and fails to resolve — the assets render backend loses the
    # RTX renderer and later crashes with CUDA illegal-memory-access errors.
    probe = (
        "import ctypes,sys;"
        f"sys.exit(0 if hasattr(ctypes.CDLL('libglib-2.0.so.0'), '{_GLIB_REQUIRED_SYMBOL}') else 1)"
    )
    try:
        if subprocess.run([sys.executable, "-c", probe], check=False,
                          capture_output=True, timeout=30).returncode == 0:
            os.environ[_GLIB_PRELOAD_SENTINEL] = "1"
            return
    except Exception:
        pass
    libs = _find_isaac_bundled_glib()
    if not libs:
        return  # nothing to preload — let the boot proceed and surface its own error
    existing = os.environ.get("LD_PRELOAD", "")
    os.environ["LD_PRELOAD"] = ":".join(libs) + (f":{existing}" if existing else "")
    os.environ[_GLIB_PRELOAD_SENTINEL] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])  # never returns


def boot_isaac_if_available(*, headless: bool = True, enable_cameras: bool = True):
    """Boot IsaacLab's ``AppLauncher`` when importable, else return ``None``.

    Applies :func:`ensure_isaac_glib_preload` first (may ``os.execv`` and not
    return). Returns the ``simulation_app`` handle (call ``.close()`` at shutdown)
    or ``None`` when IsaacLab is absent (the pure-Python path, e.g. CI).
    """
    if importlib.util.find_spec("isaaclab") is None:
        return None
    ensure_isaac_glib_preload()
    from isaaclab.app import AppLauncher

    # AppLauncher wants a dict (it calls ``.pop()`` and fills missing keys).
    return AppLauncher({"headless": headless, "enable_cameras": enable_cameras}).app
