# Policies

A policy is evaluated through an adapter, selected with `--model-type`, and a checkpoint, passed with `--checkpoint`. Adapters are in [`navsafe/policy`](../navsafe/policy); the model code they load is in [`navsafe/modelzoo`](../navsafe/modelzoo).

```bash
navsafe benchmark "<model-type>" "$NAVSAFE_DATA_ROOT/model_zoo/<checkpoint>"
```

Weights are in the dataset's `model_zoo/` directory, one subdirectory per model. They are third-party releases, each under its own license; see `model_zoo/LICENSES.md` in the dataset. Download only the models you evaluate:

```bash
hf download c13752hz/NavSafe --repo-type dataset --local-dir "$NAVSAFE_DATA_ROOT" \
  --include "model_zoo/<model-directory>/*"
```

Keep each model directory intact. Several adapters find companion files, such as anchors and vocabularies, next to the checkpoint.

## Model zoo

| `--model-type` | Checkpoint, relative to `model_zoo/` | Notes |
| :--- | :--- | :--- |
| `drivor` | `drivor/drivor_Nav1_25epochs.pth` | |
| `transfuser` | `transfuser/transfuser_seed_0.ckpt` | |
| `ltf` | `ltf/ltf_seed_0.ckpt` | Latent TransFuser; camera only. |
| `ego_mlp` | `ego_status_mlp/ego_status_mlp_seed_0.ckpt` | Ego-status baseline without camera input. |
| `diffusiondrive` | `diffusiondrive/diffusiondrive_navsim_88p1_PDMS` | Reads the anchors `kmeans_navsim_traj_20.npy` beside the checkpoint, or from `--plan-anchor-path`. |
| `sparsedrivev2` | `sparsedrivev2/sparsedrive_navsimv1_92p2.ckpt` | Reads its vocabulary files beside the checkpoint. Builds a CUDA operator on first use. |
| `gtrs_dense` | `gtrs_dense/gtrs_dense_vov.ckpt` | Also runs the SimScale checkpoints in the same directory (`gtrs_dense_{resnet,vov}_sim_{expert,reward}_navhard.ckpt`). Reads the trajectory vocabulary `16384.npy` from the same directory. |
| `recogdrive` | `recogdrive/planner_il/ReCogDrive_Diffusion_Planner_2B_IL.ckpt` | `planner_rl/` holds the RL-finetuned planner. The shared VLM is `recogdrive/vlm2b`; override with `NAVSAFE_RECOGDRIVE_VLM`. |
| `mtdrive` | `mtdrive/mtdrive_sft` | The checkpoint is a directory; `mtdrive_rl_best` is the RL stage. |
| `autovla` | `autovla/AutoVLA_PDMS_89.ckpt` | Needs `_base/qwen2.5-vl-3b-instruct`. |
| `simwam` | `simwam/weights/SimWAM-RL.pt` | Needs `_base/diffsynth`. |
| `drivelaw` | `drivelaw/inference_front.template.yaml` | The checkpoint is a configuration template naming the video model and the planner. |
| `drivevla_w0` | `drivevla_w0/Emu3_Flow_Matching_Action_Expert_PDMS_87.2` | Needs `_base/emu3-stage1` and `_base/emu3-visiontokenizer`. |
| `resworld` | `resworld/resworld_trained.pth` | Runs in its own Python environment. |

Planners without weights take `--checkpoint none`: `pdm_closed` and `idm_centerline`.

## Models that run outside the NavSafe environment

The larger vision-language policies (`autovla`, `mtdrive`, `recogdrive`, `simwam`, `drivelaw`, `drivevla_w0`) and `resworld` need dependencies that conflict with the simulator's. Their adapters start the model in a separate Python interpreter and exchange observations and plans with it.

| Variable | Meaning |
| :--- | :--- |
| `NAVSAFE_VLA_PYTHON` | Interpreter of the environment with the model's dependencies. Default `~/.cache/navsafe/venvs/vla/bin/python`. |
| `NAVSAFE_VLA_GPU` | GPU for the model process, if it should differ from the evaluator's. |
| `NAVSAFE_AUTOVLA_REPO`, `NAVSAFE_SIMWAM_REPO`, `NAVSAFE_DRIVELAW_REPO`, `NAVSAFE_DRIVEVLA_W0_REPO`, `NAVSAFE_RESWORLD_REPO` | Checkout of the model's upstream repository. Default `~/.cache/navsafe/repos/<Name>`. |
| `NAVSAFE_RESWORLD_PYTHON` | Interpreter of ResWorld's environment. |

These models use considerably more GPU memory than the camera-only policies; give them a GPU of their own.

## Model options

Most adapters need only the checkpoint. These read additional settings from the environment:

| Model | Variable | Default | Meaning |
| :--- | :--- | :--- | :--- |
| `gtrs_dense` | `NAVSAFE_GTRS_VOCAB_SIZE` | `16384` | Number of candidate trajectories, `16384` or `8192`. Results with different sizes are not comparable. |
| `gtrs_dense` | `NAVSAFE_GTRS_VOCAB` | `<model zoo>/gtrs_dense/<size>.npy` | Vocabulary file, if it is elsewhere. |
| `gtrs_dense` | `NAVSAFE_GTRS_BACKBONE` | `auto` | `resnet` or `vov`; by default detected from the checkpoint. |
| `recogdrive` | `NAVSAFE_RECOGDRIVE_VLM` | `<model zoo>/recogdrive/vlm2b` | Vision-language model shared by both planner stages. |
| `diffusiondrive` | `--plan-anchor-path` (option) | beside the checkpoint | Trajectory anchors. |

## Adding a policy

Subclass the sensor or state policy base class in `navsafe/policy`, register it with `@register_policy("<name>")`, and pass that name as `--model-type`. The existing adapters in `navsafe/policy/sensor` show how camera images, ego state and the route are converted to a model's input and how its output becomes a trajectory.
