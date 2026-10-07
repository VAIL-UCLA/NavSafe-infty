"""Checkpoint-load preconditions shared by the sensor adapters.

Every adapter here loads third-party weights with ``strict=False``, because
some legitimately reshape the module tree first (the DrivoR residual-adapter
wrapper) and some carry optional heads. What ``strict=False`` also does is
absorb a genuine mismatch in silence — a tensor the model USES and the
checkpoint does not supply runs randomly-initialised weights and reports the
number as a policy result. That has happened in this repo: the DrivoR
checkpoint loaded for its entire recorded history with 41 unexpected tensors
(the top half of the trained scorer plus the DINOv2 register tokens), and the
only signal was a printed count nothing read.

:func:`assert_state_dict_matches` turns that count into a precondition.
``navsafe.policy.sensor.drivor`` keeps its own richer variant — it additionally
recognises checkpoints trained against the pre-fix model — so this module is
the plain form for the other adapters.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence, Tuple


def keys_not_allowed(keys: Iterable[str], allowlist: Sequence[str]) -> list[str]:
    """Sorted ``keys`` minus those covered by ``allowlist``.

    An allowlist entry matches a key exactly, or matches every key under it as
    a dotted subtree — ``"scorer"`` covers ``"scorer.layers.0.w"`` but never
    ``"scorer_v2.w"``, so an entry cannot swallow a sibling module by string
    prefix.
    """
    prefixes = tuple(a.removesuffix(".") for a in allowlist)
    return sorted(
        k for k in keys if not any(k == p or k.startswith(p + ".") for p in prefixes)
    )


def sample_keys(keys: Sequence[str], limit: int = 10) -> str:
    """``limit`` keys with a count of the remainder — full lists are useless."""
    head = ", ".join(keys[:limit])
    return head + (f", … (+{len(keys) - limit} more)" if len(keys) > limit else "")


def assert_state_dict_matches(
    model: Any,
    state_dict: Dict[str, Any],
    *,
    model_name: str,
    checkpoint_path: str,
    allowed_missing: Sequence[str] = (),
    allowed_unexpected: Sequence[str] = (),
    hint: str = "",
) -> Tuple[list[str], list[str]]:
    """``load_state_dict(strict=False)``, then raise on any unwaived mismatch.

    Returns the raw ``(missing, unexpected)`` lists so a caller can log them.

    Raises:
        RuntimeError: naming the offending keys, and ``hint`` when given — the
            hint is where an adapter says which config flag usually explains
            the mismatch.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    blocked_missing = keys_not_allowed(missing, allowed_missing)
    blocked_unexpected = keys_not_allowed(unexpected, allowed_unexpected)

    if not blocked_missing and not blocked_unexpected:
        waived = (len(missing) - len(blocked_missing)) + (
            len(unexpected) - len(blocked_unexpected)
        )
        note = f", {waived} allowlisted" if waived else ""
        print(
            f"Checkpoint loaded: {len(state_dict)} tensors, {len(missing)} missing, "
            f"{len(unexpected)} unexpected{note}"
        )
        return list(missing), list(unexpected)

    lines = [
        f"{model_name} checkpoint does not match the constructed model: "
        f"{len(blocked_missing)} missing, {len(blocked_unexpected)} unexpected "
        f"(checkpoint {checkpoint_path}).",
    ]
    if blocked_missing:
        lines.append(
            "  MISSING (the model uses these; the checkpoint does not supply "
            f"them, so they stay randomly initialised): {sample_keys(blocked_missing)}"
        )
    if blocked_unexpected:
        lines.append(
            "  UNEXPECTED (the checkpoint trained these; the model has no such "
            f"tensors, so they are discarded): {sample_keys(blocked_unexpected)}"
        )
    if hint:
        lines.append(f"  {hint}")
    raise RuntimeError("\n".join(lines))


__all__ = ["assert_state_dict_matches", "keys_not_allowed", "sample_keys"]
