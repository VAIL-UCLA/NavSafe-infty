"""Strict original-track appearance contract for BEV-authored takeovers."""

def required_harvest_tracks(scenario):
    return {str(row['track_id']) for row in scenario.get('metadata', {}).get('navsafe_reactive', [])
            if row.get('require_harvested_asset')}


def select_takeover_replacements(doc, plys, scene_tracks, scene_ids, cap, required):
    required = set(required)
    missing = required - set(plys)
    if missing:
        raise ValueError(f"Takeover has no original harvested asset in manifest: {sorted(missing)}")
    # A 20 s handoff is four independently reconstructed 5 s scenes.  A logged
    # actor commonly exists in only some of them; absence from another window
    # means there is no baked actor there to replace.  What must never happen is
    # silently selecting a manifest that does not contain the takeover in any
    # served scene.
    served = set().union(*(set(scene_tracks.get(sid, ())) for sid in scene_ids))
    missing = required - served
    if missing:
        raise ValueError(
            f"Takeover track absent from every controllable scene: {sorted(missing)}")
    cost = lambda tid: sum(tid in scene_tracks.get(sid, ()) for sid in scene_ids)
    used = sum(cost(t) for t in required)
    if cap > 0 and used > cap:
        raise ValueError(f"Required takeover tracks need {used} instances, over harvester budget {cap}")
    if cap <= 0:
        return dict(plys)
    kept = set(required)
    near = sorted((t for t in plys if t not in required),
                  key=lambda t: (doc['assets'][t].get('min_ego_dist_m') if doc['assets'][t].get('min_ego_dist_m') is not None else float('inf'),t))
    for tid in near:
        n = cost(tid)
        if not n:
            continue
        if used+n > cap:
            break
        kept.add(tid); used+=n
    return {t:p for t,p in plys.items() if t in kept}
