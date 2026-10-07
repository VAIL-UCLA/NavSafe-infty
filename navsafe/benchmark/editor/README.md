# Event editor

A browser tool for authoring controlled events on a bird's-eye view of a scenario: a vehicle cutting in, a lead vehicle braking, or a pedestrian or animal crossing. It produced the recipes of the [state-perturbation set](../recipes/proxy_set_state_perturbation/README.md).

The editor writes recipes. It does not render the scenario or evaluate a policy, and its geometry checks do not establish that an event is valid for evaluation.

## Start

```bash
navsafe editor \
  --inventory "<inventory-json>" --output "<editor-output-directory>" \
  --recipe-dir "<recipe-directory>" --port "<editor-port>"
```

| Option | Meaning |
| :--- | :--- |
| `--inventory` | JSON list of the scenarios to edit. Each entry has `token`, `leaf` (event-type id), `scene_id` and `data_root`, the absolute path of the bundle's `arrow/` directory. Required. |
| `--output` | Directory for drafts, exported recipes and check reports. Required. |
| `--recipe-dir` | Directory with the scenarios' existing recipes, whose actors are shown and preserved. Required. |
| `--port` | Local HTTP port, default `8765`. |

Set `NAVSAFE_DATA_ROOT` to the dataset root, then open `http://127.0.0.1:<editor-port>`. The server listens on the loopback interface only; to use it on a remote machine or in a Pod, forward the port.

## Authoring an event

1. Choose a scenario and an event tab.
2. Pick an actor type and click on the view, or drag the type onto it. For braking and cut-in events, clicking a logged vehicle offers to take that vehicle over instead of inserting a new one.
3. The orange point is the actor's position when the event starts. Drag it to move the whole path, and drag the white points to shape the path. The path can be extended by 50 m or trimmed by one point.
4. Set the start time, speed, direction and braking parameters, and play the timeline. Before the event starts, a vehicle approaches along the path's initial direction and a pedestrian or animal waits at the start.
5. Check the geometry and export.

An export contains `event.yaml`, `baseline.yaml`, an editable `draft.json` and `checks.json`. The baseline has the same actor with the event switched off. Actors of the scenario's existing recipe and its warm-up length are kept.

## Limits

- Coordinates are metres in the frame of the ego's first pose. New paths are placed at height zero, so check the vertical placement in a render.
- A vehicle whose path runs out fails the export; a pedestrian or animal stops at the end of its path.
- The geometry check samples the event every 0.2 s against the logged ego and traffic. An intersection with the logged ego marks a potential hazard, not the outcome of a closed-loop run.
