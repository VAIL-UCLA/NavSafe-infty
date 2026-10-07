# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""``navsafe doctor`` — diagnose the NavSafe environment without booting kit.

Checks the environment requirements documented in
``docs/installation.md`` and prints one PASS / WARN / FAIL line per check,
each with the exact remediation command. Exits non-zero iff any check FAILs.

Design constraints:

* **Never boots IsaacSim.** Presence/version checks read package *metadata*
  (:func:`importlib.metadata.version`) and module specs
  (:func:`importlib.util.find_spec`) — no ``import isaacsim``, no
  ``SimulationApp``.
* **Degrades gracefully.** On a machine without the heavy stack (e.g. the
  stock CI runner) the missing pieces are reported as FAIL/WARN lines with
  install directions — never a traceback.
* **Testable.** All environment access goes through a :class:`Probe` of
  injectable callables so tests can simulate healthy and broken machines.

The checks (see ``docs/installation.md`` for the rationale behind each pin):

==================  ==========================================================
python-version      Python must be 3.12 (IsaacSim 6.0 ships cp312 wheels only)
numpy-pin           ``numpy==2.3.1`` (IsaacSim 6.0 ABI; IsaacLab's URDF
                    importers silently downgrade it)
warp-pin            ``warp-lang==1.12.0`` — a newer pip warp shadows IsaacSim's
                    bundled warp, replicator fails to load, and the ``assets``
                    render backend silently captures BLACK frames
websockets-pin      ``websockets==12.0`` (IsaacSim kernel / viser override)
isaacsim            installed and version 6.x
isaaclab            importable (source-installed, NOT in ``uv.lock``)
uv-no-sync          ``UV_NO_SYNC=1`` guard so a plain ``uv sync``/``uv run``
                    does not remove the source-installed IsaacLab
eula                ``ACCEPT_EULA`` / ``OMNI_KIT_ACCEPT_EULA`` exported
conda-ld-preload    ``LD_PRELOAD`` hint when conda is on PATH (GLIBCXX_3.4.30
                    kit bootstrap abort)
cuda-visibility     GPU present; ``CUDA_VISIBLE_DEVICES`` pins one device on
                    multi-GPU hosts (policy and IsaacSim must share a device)
py123d-data-root    ``PY123D_DATA_ROOT``, when set, points at an existing dir
==================  ==========================================================
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Mapping, Optional, Sequence

_DOCS = "docs/installation.md"

# Documented pins (single place; keep in sync with pyproject.toml).
_NUMPY_PIN = "2.3.1"
_WARP_PIN = "1.12.0"
_WEBSOCKETS_PIN = "12.0"
_ISAACSIM_MAJOR = "6"

_PASS, _WARN, _FAIL = "PASS", "WARN", "FAIL"


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single doctor check."""

    name: str
    status: str  # PASS | WARN | FAIL
    detail: str
    remediation: str = ""


def _real_dist_version(dist: str) -> Optional[str]:
    """Installed version of a distribution, or ``None`` — metadata only, no import."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 — a broken dist dir must not crash doctor
        return None


def _real_module_present(name: str) -> bool:
    """Whether ``import name`` would find a module — spec lookup only, no import."""
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _real_gpu_names() -> Optional[List[str]]:
    """GPU names from ``nvidia-smi -L``; ``None`` when no driver/CLI is present."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        out = subprocess.run(
            [smi, "-L"], capture_output=True, text=True, check=False, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    # Count only physical GPUs ("GPU 0: ..."); on MIG-enabled hosts
    # `nvidia-smi -L` also emits indented "MIG ..." instance lines.
    return [
        ln.strip() for ln in out.stdout.splitlines()
        if ln.strip().startswith("GPU ")
    ]


@dataclass
class Probe:
    """Injectable environment access for :func:`run_checks` (mock in tests)."""

    env: Mapping[str, str] = field(default_factory=lambda: os.environ)
    python_version: tuple = field(default_factory=lambda: tuple(sys.version_info[:3]))
    dist_version: Callable[[str], Optional[str]] = _real_dist_version
    module_present: Callable[[str], bool] = _real_module_present
    which: Callable[[str], Optional[str]] = shutil.which
    gpu_names: Callable[[], Optional[List[str]]] = _real_gpu_names
    is_dir: Callable[[str], bool] = os.path.isdir


def _version_tuple(v: str) -> tuple:
    """Loose numeric-prefix parse for ordering (``'1.15.0.dev3'`` → (1, 15, 0))."""
    parts: List[int] = []
    for tok in v.split("."):
        digits = ""
        for ch in tok:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


# ---------------------------------------------------------------------------
# Individual checks. Each takes the probe and returns one CheckResult.
# ---------------------------------------------------------------------------


def _check_python(p: Probe) -> CheckResult:
    got = ".".join(str(x) for x in p.python_version[:3])
    if p.python_version[:2] == (3, 12):
        return CheckResult("python-version", _PASS, f"Python {got}")
    return CheckResult(
        "python-version", _FAIL,
        f"Python {got} — NavSafe requires 3.12 (IsaacSim 6.0 ships cp312 wheels only)",
        f"uv sync --all-extras --python 3.12   (see {_DOCS})",
    )


def _check_numpy(p: Probe) -> CheckResult:
    v = p.dist_version("numpy")
    if v is None:
        return CheckResult(
            "numpy-pin", _FAIL, "numpy is not installed",
            f"uv sync --all-extras --python 3.12   (see {_DOCS})",
        )
    if v == _NUMPY_PIN:
        return CheckResult("numpy-pin", _PASS, f"numpy {v}")
    return CheckResult(
        "numpy-pin", _FAIL,
        f"numpy {v} != {_NUMPY_PIN} (IsaacSim 6.0 ABI; IsaacLab's nvidia-srl-usd* "
        "importers are the usual culprit for a silent downgrade)",
        f'uv pip install "numpy=={_NUMPY_PIN}"   (see {_DOCS} install step 3)',
    )


def _check_warp(p: Probe) -> CheckResult:
    v = p.dist_version("warp-lang")
    if v is None:
        # Not having a pip warp at all is safe: IsaacSim uses its bundled
        # omni.warp.core. Only a *different* pip version shadows it.
        return CheckResult(
            "warp-pin", _PASS,
            "no pip warp-lang installed (IsaacSim's bundled warp is used)",
        )
    if v == _WARP_PIN:
        return CheckResult("warp-pin", _PASS, f"warp-lang {v}")
    if _version_tuple(v) > _version_tuple(_WARP_PIN):
        detail = (
            f"warp-lang {v} > {_WARP_PIN}: a newer pip warp SHADOWS IsaacSim's "
            "bundled warp — replicator fails to load and the `assets` render "
            "backend silently captures BLACK frames"
        )
    else:
        detail = f"warp-lang {v} != {_WARP_PIN} (pin required by omni.replicator.core)"
    return CheckResult(
        "warp-pin", _FAIL, detail,
        f'uv pip install "warp-lang=={_WARP_PIN}"   (see {_DOCS} / pyproject [tool.uv])',
    )


def _check_websockets(p: Probe) -> CheckResult:
    v = p.dist_version("websockets")
    if v is None:
        return CheckResult(
            "websockets-pin", _WARN,
            "websockets is not installed (required by the IsaacSim kernel)",
            f"uv sync --all-extras --python 3.12   (see {_DOCS})",
        )
    if v == _WEBSOCKETS_PIN:
        return CheckResult("websockets-pin", _PASS, f"websockets {v}")
    return CheckResult(
        "websockets-pin", _FAIL,
        f"websockets {v} != {_WEBSOCKETS_PIN} (IsaacSim kernel hard-pins 12.0; "
        "the uv override resolves the viser conflict — something re-resolved it)",
        f'uv pip install "websockets=={_WEBSOCKETS_PIN}"   (see pyproject '
        "[tool.uv] override-dependencies)",
    )


def _check_isaacsim(p: Probe) -> CheckResult:
    v = p.dist_version("isaacsim")
    if v is None:
        if p.module_present("isaacsim"):
            return CheckResult(
                "isaacsim", _WARN,
                "`isaacsim` module found but no `isaacsim` distribution metadata — "
                "cannot verify the 6.x pin",
                f"uv sync --all-extras --python 3.12   (see {_DOCS})",
            )
        return CheckResult(
            "isaacsim", _FAIL,
            "IsaacSim is not installed (heavy stack absent — expected on a "
            "CI/laptop checkout, required on sim machines)",
            f"uv sync --all-extras --python 3.12   (see {_DOCS}; IsaacSim 6.0.0 "
            "resolves from https://pypi.nvidia.com via pyproject [tool.uv])",
        )
    if v.split(".")[0] == _ISAACSIM_MAJOR:
        return CheckResult("isaacsim", _PASS, f"isaacsim {v}")
    return CheckResult(
        "isaacsim", _FAIL,
        f"isaacsim {v} — NavSafe pins IsaacSim 6.0.0 (pyproject [project.dependencies])",
        f"uv sync --all-extras --python 3.12   (see {_DOCS})",
    )


def _check_isaaclab(p: Probe) -> CheckResult:
    if p.module_present("isaaclab"):
        return CheckResult("isaaclab", _PASS, "isaaclab importable (source install)")
    return CheckResult(
        "isaaclab", _FAIL,
        "IsaacLab is not importable. It is source-installed (NOT in uv.lock) — "
        "a plain `uv sync` / `uv run` removes it",
        "git clone --branch v3.0.0-beta https://github.com/isaac-sim/IsaacLab.git "
        "../IsaacLab && (cd ../IsaacLab && ./isaaclab.sh -i); then restore numpy: "
        f'uv pip uninstall nvidia-srl-usd nvidia-srl-usd-to-urdf && uv pip install '
        f'"numpy=={_NUMPY_PIN}"   (see {_DOCS} steps 2-3)',
    )


def _check_uv_no_sync(p: Probe) -> CheckResult:
    if not p.module_present("isaaclab"):
        # Without a source-installed IsaacLab there is nothing for a plain
        # `uv sync` to remove; the isaaclab check above already reports it.
        return CheckResult(
            "uv-no-sync", _PASS, "n/a (no source-installed IsaacLab to protect)"
        )
    if p.env.get("UV_NO_SYNC", "").lower() in ("1", "true", "yes", "on"):
        return CheckResult(
            "uv-no-sync", _PASS, f"UV_NO_SYNC={p.env['UV_NO_SYNC']}"
        )
    return CheckResult(
        "uv-no-sync", _WARN,
        "UV_NO_SYNC is not set: a plain `uv sync` / `uv run` reconciles the env "
        "to uv.lock and would REMOVE the source-installed IsaacLab",
        "export UV_NO_SYNC=1   (or always `uv run --no-sync ...`; re-sync with "
        f"`uv sync --inexact` — see {_DOCS} install notes)",
    )


def _check_eula(p: Probe) -> CheckResult:
    # Accepting values only — ACCEPT_EULA=N is set but does NOT accept.
    accepting = ("y", "yes", "1", "true")
    bad = [
        var for var in ("ACCEPT_EULA", "OMNI_KIT_ACCEPT_EULA")
        if p.env.get(var, "").lower() not in accepting
    ]
    if not bad:
        return CheckResult(
            "eula", _PASS, "ACCEPT_EULA and OMNI_KIT_ACCEPT_EULA accept"
        )
    return CheckResult(
        "eula", _WARN,
        f"{' and '.join(bad)} not set to an accepting value — anything that "
        "boots IsaacSim (simulation, evaluation scripts, smoke "
        "renders) will refuse or hang",
        "export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES",
    )


def _check_conda_ld_preload(p: Probe) -> CheckResult:
    conda = p.which("conda") or p.env.get("CONDA_EXE")
    if not conda:
        return CheckResult("conda-ld-preload", _PASS, "conda not on PATH")
    if "libstdc++" in p.env.get("LD_PRELOAD", ""):
        return CheckResult(
            "conda-ld-preload", _PASS,
            "conda on PATH and LD_PRELOAD already pins the system libstdc++",
        )
    return CheckResult(
        "conda-ld-preload", _WARN,
        "conda is on PATH without an LD_PRELOAD guard: conda's libstdc++ can "
        "shadow the system one and abort the kit bootstrap with "
        "`GLIBCXX_3.4.30 not found`",
        "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6   "
        f"(see {_DOCS} troubleshooting)",
    )


def _check_cuda_visibility(p: Probe) -> CheckResult:
    cvd = p.env.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() in ("", "-1"):
        return CheckResult(
            "cuda-visibility", _FAIL,
            f"CUDA_VISIBLE_DEVICES={cvd!r} hides every GPU from this process",
            "export CUDA_VISIBLE_DEVICES=0",
        )
    gpus = p.gpu_names()
    if gpus is None:
        return CheckResult(
            "cuda-visibility", _WARN,
            "nvidia-smi unavailable or failed (missing or unhealthy NVIDIA "
            "driver?) — fine for light dev, but every IsaacSim-backed "
            "workflow needs an NVIDIA GPU (driver 535+)",
            "install/repair the NVIDIA driver (535+) on sim machines",
        )
    if len(gpus) == 0:
        return CheckResult(
            "cuda-visibility", _WARN, "nvidia-smi reports no GPUs",
            "check the NVIDIA driver (535+) and that a GPU is attached",
        )
    if len(gpus) > 1 and cvd is None:
        return CheckResult(
            "cuda-visibility", _WARN,
            f"{len(gpus)} GPUs visible and CUDA_VISIBLE_DEVICES unset — the "
            "policy and IsaacSim must share a device or renders come back empty",
            "export CUDA_VISIBLE_DEVICES=0   (pick one device)",
        )
    pinned = f", CUDA_VISIBLE_DEVICES={cvd}" if cvd is not None else ""
    return CheckResult("cuda-visibility", _PASS, f"{len(gpus)} GPU(s){pinned}")


def _check_py123d_data_root(p: Probe) -> CheckResult:
    root = p.env.get("PY123D_DATA_ROOT")
    if root is None:
        return CheckResult(
            "py123d-data-root", _PASS,
            "PY123D_DATA_ROOT not set (only needed for py123d-backed eval; "
            "scripts also accept --py123d-data-root)",
        )
    if p.is_dir(root):
        return CheckResult("py123d-data-root", _PASS, f"PY123D_DATA_ROOT={root}")
    return CheckResult(
        "py123d-data-root", _FAIL,
        f"PY123D_DATA_ROOT={root} does not exist",
        "point PY123D_DATA_ROOT at your converted Arrow root (see "
        "docs/data_preparation.md) or unset it",
    )


_CHECKS = (
    _check_python,
    _check_numpy,
    _check_warp,
    _check_websockets,
    _check_isaacsim,
    _check_isaaclab,
    _check_uv_no_sync,
    _check_eula,
    _check_conda_ld_preload,
    _check_cuda_visibility,
    _check_py123d_data_root,
)


def run_checks(probe: Optional[Probe] = None) -> List[CheckResult]:
    """Run every doctor check; a crashing check becomes its own FAIL line."""
    p = probe if probe is not None else Probe()
    results: List[CheckResult] = []
    for check in _CHECKS:
        try:
            results.append(check(p))
        except Exception as exc:  # noqa: BLE001 — doctor must never traceback
            results.append(CheckResult(
                check.__name__.removeprefix("_check_").replace("_", "-"),
                _FAIL, f"check crashed: {exc!r}",
                "report this at the NavSafe issue tracker",
            ))
    return results


def format_report(results: Sequence[CheckResult]) -> str:
    """Human-readable report: one status line per check + a summary line."""
    lines: List[str] = ["navsafe doctor — environment invariants "
                        f"(see {_DOCS})", ""]
    width = max(len(r.name) for r in results)
    for r in results:
        lines.append(f"[{r.status}] {r.name.ljust(width)}  {r.detail}")
        if r.remediation and r.status != _PASS:
            lines.append(f"{' ' * (width + 9)}fix: {r.remediation}")
    n_fail = sum(r.status == _FAIL for r in results)
    n_warn = sum(r.status == _WARN for r in results)
    n_pass = sum(r.status == _PASS for r in results)
    lines.append("")
    lines.append(f"{n_pass} passed, {n_warn} warned, {n_fail} failed")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``navsafe doctor``. Exits 1 iff any check FAILs."""
    parser = argparse.ArgumentParser(
        prog="navsafe doctor",
        description=(
            "Diagnose the NavSafe environment: version pins, IsaacSim/IsaacLab "
            "presence, EULA/CUDA env vars. Never boots IsaacSim."
        ),
    )
    parser.parse_args(argv)
    results = run_checks()
    print(format_report(results))
    return 1 if any(r.status == _FAIL for r in results) else 0


if __name__ == "__main__":  # pragma: no cover — exercised via console script
    raise SystemExit(main())
