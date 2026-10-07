"""Validate completed NavSafe episode outputs without requiring visualization."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_episode(directory: str | Path, *, require_vis: bool = False) -> dict:
    """Check final scoring output; excluded episodes do not count as completed evals."""
    root = Path(directory)
    result = {"directory": str(root), "status": None, "passed": False, "errors": []}
    errors = result["errors"]
    try:
        data = json.loads((root / "navsafe_metrics.json").read_text())
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
    except (OSError, ValueError) as exc:
        errors.append(f"navsafe_metrics.json: {exc}")
        return result
    result["status"] = data.get("status")
    if result["status"] != "scored":
        errors.append(f"episode status is {result['status']!r}, expected 'scored'")
    frames = data.get("frames")
    if not isinstance(frames, dict):
        errors.append("missing frames object")
    else:
        counts = [frames.get(k) for k in ("total", "scored", "warmup_excluded")]
        if not all(type(v) is int and v >= 0 for v in counts):
            errors.append("frame counts must be nonnegative integers")
        elif counts[1] == 0 or counts[1] + counts[2] > counts[0]:
            errors.append("empty or inconsistent scored frame window")
    termination = data.get("termination")
    if not isinstance(termination, dict) or not isinstance(termination.get("reason"), str) or not termination["reason"]:
        errors.append("missing termination reason")
    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        errors.append("missing metrics object")
    elif result["status"] == "scored":
        if not _finite(metrics.get("driving_score")):
            errors.append("driving_score must be finite (zero is valid)")
        if type(metrics.get("success")) is not bool:
            errors.append("success must be a boolean")
        for key in ("efficiency_pct", "comfort"):
            if key not in metrics or (metrics[key] is not None and not _finite(metrics[key])):
                errors.append(f"{key} must be present and finite or null")
    # Image resolution and compression vary by policy. Decode actual files,
    # rather than enforcing a particular byte size or camera resolution.
    images = sorted((root / "frames").rglob("*.jpg")) + sorted((root / "frames").rglob("*.png"))
    gifs = sorted((root / "visualization").glob("*.gif"))
    if require_vis:
        for name in ("cam_f0.jpg", "topdown.png"):
            if not any(p.name == name for p in images):
                errors.append(f"visualization requested but {name} is missing")
        for name in ("cam_f0.gif", "topdown.gif"):
            if not any(p.name == name for p in gifs):
                errors.append(f"visualization requested but {name} is missing")
    if images or gifs:
        from PIL import Image
        for path in images + gifs:
            try:
                with Image.open(path) as image:
                    for index in range(getattr(image, "n_frames", 1)):
                        image.seek(index)
                        image.load()
            except (OSError, ValueError, EOFError) as exc:
                errors.append(f"invalid image {path.relative_to(root)}: {exc}")
    result["passed"] = not errors
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("scenario_dir", nargs="?", type=Path, help="One episode output directory")
    source.add_argument("--model-dir", type=Path, help="Recursively validate episodes under this directory")
    parser.add_argument("--require-vis", action="store_true", help="Require front-camera/BEV frames and GIFs")
    args = parser.parse_args(argv)
    if args.model_dir is not None:
        roots = {p.parent for name in ("navsafe_metrics.json", "metrics.json", "vehicle_states.npy", "eval.log")
                 for p in args.model_dir.rglob(name)}
        directories = sorted(roots)
    else:
        directories = [args.scenario_dir]
    if not directories:
        print("No episode outputs found")
        return 1
    results = [validate_episode(p, require_vis=args.require_vis) for p in directories]
    print(json.dumps(results, indent=2))
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
