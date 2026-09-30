# Fixed-28 BEV event editor

Use the fixed proxy set to author cut-in, lead braking and pedestrian/animal crossings. This tool does not replace proxy scenarios or qualify an event as safe/valid for evaluation.

## Start locally or in your own evaluation pod

Activate the environment installed using the repository README.
Set NAVSAFE_DATA_ROOT to the downloaded HF snapshot. Supply an inventory JSON
with the 28 proxy tokens, leaf, scene_id and data_root (the absolute path to each
bundle's arrow directory). The inventory and optional cached scenario objects
are local working files, not public dataset paths.

```bash
python -m navsafe.benchmark.editor.server \
  --inventory "<inventory-json>" --output "<editor-output-directory>" \
  --recipe-dir "<frozen-recipe-directory>" --port "<editor-port>"
```

- `-m navsafe.benchmark.editor.server` — **required invocation**; starts the local editor service.
- `--inventory` — **required**; JSON inventory containing the proxy tokens, leaves, scene IDs and absolute Arrow data roots.
- `--output` — **required**; directory where drafts, exported recipes and geometry-check reports are written.
- `--recipe-dir` — **required**; directory containing the matching original frozen recipes.
- `--port` — **optional**, default `8765`; local HTTP port. Choose an unused port for each editor instance.


Open `http://127.0.0.1:<editor-port>` locally. For a pod, forward that port using your own
namespace and pod name. The editor binds loopback and creates no public Service.

## Interaction

1. Choose one of the 28 scenarios and an event tab.
2. Select an actor type and click BEV, or drag the type onto BEV. For braking/cut-in, clicking a source vehicle also offers takeover.
3. The orange point is the actor position **at event onset**. Drag it to move the whole path; drag white points to shape the path. Edit path appends points, delete-last-point trims it, extend adds 50 m.
4. Set onset, speed, direction, and braking parameters; scrub/play the timeline. Before onset, vehicles follow the initial tangent backwards from the placed point; VRUs wait there.
5. Check geometry and export. Each export contains `event.yaml`, `baseline.yaml`, editable `draft.json`, and `checks.json`. Baseline has the same actor and prefix, with no braking/cut-in/crossing. Original recipe actors and handoff remain intact.

All coordinates are ego-frame-0 metres. New paths currently use z=0; vertical placement and animation require render validation. Cars fail explicitly if the path is exhausted; VRUs stop at the endpoint. For cut-in baseline, the actor continues on the initial tangent; curved-road baselines may need a future independently editable continuation.

Geometry checks sample the event branch every 0.2 s against logged ego/traffic and previewed original recipe actors. An ego intersection is a potential hazard, not a closed-loop collision result. Checks do not guarantee physical feasibility, event observability, baseline validity, or render correctness. Original reactive actor previews use logged ego, so they can change during actual closed-loop driving.

Draft count is authoring progress, not qualification progress. UI smoke-test exports are drafts as well. Test report: `output/proxy28_bev_editor/verification.json`.
