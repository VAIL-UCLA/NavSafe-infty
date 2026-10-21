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

A model whose notes below name a `_base/` directory also needs that shared base model. Download it the same way, for example `--include "model_zoo/_base/emu3-stage1/*"`.

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
| `mtdrive` | `mtdrive/mtdrive_sft` | The checkpoint is a directory; `mtdrive_rl_best` is the RL stage. The inference code follows the MTDrive paper and will be updated when the authors release theirs. |
| `autovla` | `autovla/AutoVLA_PDMS_89.ckpt` | Needs `_base/qwen2.5-vl-3b-instruct`. |
| `simwam` | `simwam/weights/SimWAM-RL.pt` | Needs `_base/diffsynth`. |
| `drivelaw` | `drivelaw/inference_front.template.yaml` | The checkpoint is a configuration template naming the video model and the planner. The adapter fills its placeholders with the template's directory and the DriveLaW checkout. |
| `drivevla_w0` | `drivevla_w0/Emu3_Flow_Matching_Action_Expert_PDMS_87.2` | Needs `_base/emu3-stage1` and `_base/emu3-visiontokenizer`. |
| `resworld` | `resworld/resworld_trained.pth` | Runs in its own Python environment. |

Planners without weights take `--checkpoint none`: `pdm_closed` and `idm_centerline`.

## Models that run outside the NavSafe environment

The larger vision-language policies (`autovla`, `mtdrive`, `recogdrive`, `simwam`, `drivelaw`, `drivevla_w0`) and `resworld` need dependencies that conflict with the simulator's. Their adapters start the model in a separate Python interpreter and exchange observations and plans with it.

| Variable | Meaning |
| :--- | :--- |
| `NAVSAFE_VLA_PYTHON` | Interpreter of the environment with the model's dependencies. Default `~/.cache/navsafe/venvs/vla/bin/python`. |
| `NAVSAFE_VLA_GPU` | GPU for the model process, if it should differ from the evaluator's. |
| `NAVSAFE_AUTOVLA_REPO`, `NAVSAFE_SIMWAM_REPO`, `NAVSAFE_DRIVELAW_REPO`, `NAVSAFE_DRIVEVLA_W0_REPO`, `NAVSAFE_RESWORLD_REPO` | Checkout of the model's upstream repository. Default: its submodule in `third_party/`. |
| `NAVSAFE_RESWORLD_PYTHON` | Interpreter of ResWorld's environment. Default `~/.cache/navsafe/venvs/resworld/bin/python`. |
| `NAVSAFE_VLA_EXTRA_PYTHONPATH` | Extra package directory for the `NAVSAFE_VLA_PYTHON` process. ResWorld's interpreter does not get it. The evaluator's own `PYTHONPATH` is never passed on. |

The model process runs a server script from [`navsafe/modelzoo/navsim/vla_server`](../navsafe/modelzoo/navsim/vla_server). Before the first run, fetch the model's upstream code and build its environment as described below.

### Upstream code

Five of these models import code from their upstream repository. Each repository is a git submodule in `third_party/`, pinned to the commit that NavSafe was tested with. `git clone --recurse-submodules` skips them; fetch the ones you evaluate:

```bash
git submodule update --init --checkout third_party/AutoVLA third_party/DriveLaW third_party/DriveVLA-W0 third_party/ResWorld third_party/SimWAM
```

| `--model-type` | Submodule | Upstream repository |
| :--- | :--- | :--- |
| `autovla` | `third_party/AutoVLA` | [ucla-mobility/AutoVLA](https://github.com/ucla-mobility/AutoVLA) |
| `drivelaw` | `third_party/DriveLaW` | [xiaomi-research/drivelaw](https://github.com/xiaomi-research/drivelaw) |
| `drivevla_w0` | `third_party/DriveVLA-W0` | [BraveGroup/DriveVLA-W0](https://github.com/BraveGroup/DriveVLA-W0) |
| `resworld` | `third_party/ResWorld` | [mengtan00/ResWorld](https://github.com/mengtan00/ResWorld) |
| `simwam` | `third_party/SimWAM` | [H-EmbodVis/SimWAM](https://github.com/H-EmbodVis/SimWAM) |

`recogdrive` and `mtdrive` need no upstream checkout. The ReCogDrive planner is in [`navsafe/modelzoo/navsim/recogdrive`](../navsafe/modelzoo/navsim/recogdrive), and MTDrive loads from its checkpoint directory.

### Model environment

All of these models except ResWorld share one environment. [`deploy/envs/vla`](../deploy/envs/vla) pins the versions that NavSafe was tested with:

| File | Contents |
| :--- | :--- |
| `requirements.lock.txt` | The shared environment. |
| `drivevla_w0-extra.lock.txt` | Tokenizer packages that DriveVLA-W0 needs. |
| `drivelaw-extra.lock.txt` | The nuPlan devkit and its dependencies, which DriveLaW's planner imports. They conflict with the shared environment, so they go into a separate directory. |

```bash
VLA=~/.cache/navsafe/venvs/vla
uv venv --python 3.12 "$VLA"
grep -v '^omegaconf==' deploy/envs/vla/requirements.lock.txt > /tmp/vla_requirements.txt
uv pip install --no-config --python "$VLA/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
  -r /tmp/vla_requirements.txt
uv pip install --no-config --python "$VLA/bin/python" --no-deps omegaconf==2.4.0.dev14
uv pip install --no-config --python "$VLA/bin/python" -r deploy/envs/vla/drivevla_w0-extra.lock.txt
uv pip install --no-config --target ~/.cache/navsafe/vla_extra_site --no-deps -r deploy/envs/vla/drivelaw-extra.lock.txt
export NAVSAFE_VLA_EXTRA_PYTHONPATH=~/.cache/navsafe/vla_extra_site
```

Run the block from the repository root. `--no-config` stops uv from applying the `[tool.uv]` pins in NavSafe's `pyproject.toml`, which belong to the simulator environment. `omegaconf==2.4.0.dev14` conflicts with the `omegaconf<2.4` that `hydra-core` declares, so it is installed last without dependencies. SimWAM and DriveLaW were run with this pair.

The environment must be able to compile C extensions: ReCogDrive compiles GPU kernels with `torch.compile` on its first step, which needs `gcc` and the Python headers. A Python installed by uv includes the headers; for Ubuntu's system Python, install `python3-dev`.

ResWorld needs Python 3.8 and the mmdet3d 0.17.1 stack. Install it as step 1 of "Get Started" in `third_party/ResWorld/README.md` describes: `torch==1.9.1+cu111`, `mmcv-full==1.4.0`, `mmdet==2.14.0`, `mmsegmentation==0.14.1`, mmdet3d `v0.17.1` from source and `nuscenes-devkit==1.1.9`. Then set `NAVSAFE_RESWORLD_PYTHON` to its interpreter. The environment that NavSafe was tested with has one change in mmdet3d: `mmdet3d/datasets/pipelines/data_augment_utils.py` imports `NumbaPerformanceWarning` from `numba.core.errors` instead of `numba.errors`.

### Running

These models use considerably more GPU memory than the camera-only policies. The model process runs on the simulator's GPU unless `NAVSAFE_VLA_GPU` names another one. If the simulator's GPU does not have enough free memory for the model, give the model a GPU of its own, as in this example with one GPU each for the renderer, the simulator and the model:

```bash
export NAVSAFE_RENDER_GPU=0 NAVSAFE_GPU=1 NAVSAFE_VLA_GPU=2
navsafe benchmark drivelaw "$NAVSAFE_DATA_ROOT/model_zoo/drivelaw/inference_front.template.yaml"
```

## Model options

Most adapters need only the checkpoint. These read additional settings from the environment:

| Model | Variable | Default | Meaning |
| :--- | :--- | :--- | :--- |
| `gtrs_dense` | `NAVSAFE_GTRS_VOCAB_SIZE` | `16384` | Number of candidate trajectories, `16384` or `8192`. Results with different sizes are not comparable. |
| `gtrs_dense` | `NAVSAFE_GTRS_VOCAB` | `<model zoo>/gtrs_dense/<size>.npy` | Vocabulary file, if it is elsewhere. |
| `gtrs_dense` | `NAVSAFE_GTRS_BACKBONE` | `auto` | `resnet` or `vov`; by default detected from the checkpoint. |
| `recogdrive` | `NAVSAFE_RECOGDRIVE_VLM` | `<model zoo>/recogdrive/vlm2b` | Vision-language model shared by both planner stages. |
| `autovla` | `NAVSAFE_AUTOVLA_BASE` | `<model zoo>/_base/qwen2.5-vl-3b-instruct` | Base vision-language model. |
| `simwam` | `NAVSAFE_SIMWAM_TASK` | `navsim_uncond_front_384x672_1e-4` | Upstream task that builds the model. Serves both released checkpoints. |
| `simwam` | `NAVSAFE_SIMWAM_VIDEO_DIR` | unset | Directory for the world model's predicted video, one file per replan. Much slower, and the plan differs slightly from the scored one; do not set it on a scoring run. |
| `drivevla_w0` | `NAVSAFE_DRIVEVLA_W0_VLM` | `<model zoo>/_base/emu3-stage1` | Base Emu3 model the checkpoint was fine-tuned from. |
| `drivevla_w0` | `NAVSAFE_DRIVEVLA_W0_VQ` | `<model zoo>/_base/emu3-visiontokenizer` | Emu3 vision tokenizer. |
| `drivevla_w0` | `NAVSAFE_DRIVEVLA_W0_NORM_STATS` | `<checkout>/configs/normalizer_navsim_trainval/norm_stats.json` | Action normalisation statistics. |
| `drivevla_w0` | `NAVSAFE_DRIVEVLA_W0_ACTION_TOKENIZER` | `<checkout>/configs/fast` | FAST action tokenizer. The model zoo also ships one in `drivevla_w0/fast`. |
| `drivevla_w0` | `NAVSAFE_DRIVEVLA_W0_STEPS` | `10` | Flow-matching steps. |
| `resworld` | `NAVSAFE_RESWORLD_CONFIG` | `projects/configs/resworld/resworld_config.py` | mmcv configuration; a relative path is read from the ResWorld checkout. |
| `diffusiondrive` | `--plan-anchor-path` (option) | beside the checkpoint | Trajectory anchors. |

## Adding a policy

Subclass the sensor or state policy base class in `navsafe/policy`, register it with `@register_policy("<name>")`, and pass that name as `--model-type`. The existing adapters in `navsafe/policy/sensor` show how camera images, ego state and the route are converted to a model's input and how its output becomes a trajectory.
