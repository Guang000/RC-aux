# Predictive but Not Plannable: RC-aux for Latent World Models

This repository contains the code for **Reachability-Correction auxiliary objective (RC-aux)**.  RC-aux is a lightweight training and planning correction for reconstruction-free latent world models: it keeps the LeWM backbone unchanged, trains open-loop multi-horizon prediction, adds a budget-conditioned reachability head, and optionally uses that reachability signal during planning.

<img src="./images/RC-aux.png" width="20%"/>

Paper: [arXiv:2605.07278](https://arxiv.org/abs/2605.07278)

## What Is Included

```text
.
├── train.py / eval.py        # five-task pixel-control training and MPC evaluation
├── jepa.py                   # LeWM/RC-aux latent world model
├── module.py                 # predictor, regularizer, reachability head
├── rcaux.py                  # multi-horizon and reachability objectives
├── config/
│   ├── train/                # LeWM and RC-aux configs
│   └── eval/                 # TwoRoom, Reacher, Push-T, Cube configs
├── scripts/
│   └── eval_dino_family_official.py
├── libero/                   # LIBERO-Goal OFT/BCRNN action-head scripts
├── tools/                    # fixed-group generation and success summaries
└── results/                  # result CSV summaries
```

Checkpoints and datasets are not stored in the GitHub repository.  We release the main public checkpoint separately on [Hugging Face](https://huggingface.co/biubiu116/RC-aux).

## Installation

Python 3.10 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For headless MuJoCo evaluation:

```bash
export MUJOCO_GL=egl
```

## Data and Checkpoints

The five pixel-control tasks use `stable-worldmodel`'s cache root:

```bash
export STABLEWM_HOME=/path/to/stable-wm-cache
```

Expected dataset locations:

| Task | Dataset path under `$STABLEWM_HOME` |
| --- | --- |
| TwoRoom | `tworoom.h5` |
| Reacher | `dmc/reacher_random.h5` |
| Push-T | `pusht_expert_train.h5` |
| Cube | `ogbench/cube_single_expert.h5` |
| Wall | `dino_wall.h5` |

Checkpoint paths are resolved relative to `$STABLEWM_HOME`.  Pass policy stems without the `_object.ckpt` suffix.

## Five-Task Evaluation

Example with the released TwoRoom RC-aux checkpoint:

```bash
mkdir -p "$STABLEWM_HOME/tworoom_rcaux"
cp /path/to/hf_release/rcaux/checkpoints/pixel_control/tworoom_rcaux/* "$STABLEWM_HOME/tworoom_rcaux/"

python eval.py --config-name=tworoom.yaml \
  cache_dir="$STABLEWM_HOME" \
  policy=tworoom_rcaux/rcaux_tworoom \
  +planner_override.use_reachability_cost=true \
  +planner_override.reachability_cost_weight=0.85 \
  output.filename=tworoom_rcaux_eval.txt
```

Use fixed evaluation groups by passing:

```bash
eval.row_indices_file=/path/to/group_00.json
```

Summarize logs:

```bash
python tools/summarize_group_success.py /path/to/results/*.txt
```

Wall uses the DINO-WM environment-native benchmark path.  Provide the DINO-WM source checkout and Wall dataset explicitly:

```bash
python scripts/eval_dino_family_official.py \
  --source-root /path/to/dino_wm \
  --benchmark wall \
  --policy-kind object \
  --policy /path/to/lewm_epoch_8_object.ckpt \
  --dataset-path "$STABLEWM_HOME/dino_wall.h5" \
  --val-dataset-path "$STABLEWM_HOME/dino_wall.h5" \
  --num-samples 600 \
  --n-steps 20 \
  --topk 60 \
  --horizon 8 \
  --receding-horizon 4 \
  --action-block 5 \
  --goal-cost-reduce softmin \
  --goal-cost-softmin-temperature 1.0 \
  --use-reachability-cost on \
  --reachability-cost-weight 0.85 \
  --output wall_rcaux_eval.txt
```

## Training

LeWM baseline:

```bash
python train.py --config-name=lewm.yaml \
  data=tworoom \
  wandb.enabled=false \
  output_model_name=lewm_tworoom \
  subdir=tworoom/lewm_tworoom
```

RC-aux:

```bash
python train.py --config-name=rcaux_default.yaml \
  data=tworoom \
  wandb.enabled=false \
  output_model_name=rcaux_tworoom \
  subdir=tworoom/rcaux_tworoom
```

Continuation from an existing LeWM checkpoint:

```bash
init.weights_path=/path/to/source_object.ckpt init.strict=false
```

## LIBERO-Goal

The LIBERO-Goal extension code is under `libero/`.  It trains an OFT-style action chunk head on top of the LeWM-family representation and evaluates with the official LIBERO success checker.
LIBERO-Goal does not use the MPC planner or `planner_override.*` settings; there is no `reachability_cost_weight` argument in these scripts.

Expected local layout:

```text
assets/benchmarks/LIBERO/
assets/datasets/libero_goal_agentview.h5
checkpoints/libero/lewm_epoch_40_object.ckpt
```

Train the OFT-style action head:

```bash
python libero/train_libero_goal_lewm_oft_head.py \
  --libero-root assets/benchmarks/LIBERO \
  --init-policy checkpoints/libero/lewm_epoch_40_object.ckpt \
  --tasks all \
  --image-keys agentview_rgb,eye_in_hand_rgb \
  --chunk-len 8 \
  --action-horizon 8 \
  --hidden-dim 1024 \
  --batch-size 32 \
  --max-epochs 30 \
  --train-encoder \
  --run-dir runs/libero_goal_rcaux_oft
```

Evaluate:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python libero/eval_libero_goal_lewm_oft_head.py \
  --checkpoint runs/libero_goal_rcaux_oft/lewm_libero_oft_head_epoch_30.ckpt \
  --tasks all \
  --n-eval 50 \
  --max-steps 600 \
  --output results/libero_goal_rcaux_oft_n50.json
```

On systems where robosuite cannot create an EGL offscreen context, run the same command with `MUJOCO_GL=glx`.

## Main Reported Results

The CSV summaries in `results/` mirror the reported tables.  Local LeWM-family rows are mean±std over five fixed evaluation groups of 50 episodes.

| Task | LeWM | LeWM-cont | RC-aux | Matched delta |
| --- | ---: | ---: | ---: | ---: |
| TwoRoom | 88.8±3.0 | 88.8±3.0 | 98.0±1.4 | +9.2 |
| Reacher | 81.2±7.9 | 82.8±7.2 | 87.2±6.4 | +4.4 |
| Push-T | 90.4±3.0 | 91.2±3.9 | 90.8±3.3 | -0.4 |
| Wall | 50.4±6.5 | -- | 83.6±3.6 | +33.2 |
| Cube | 72.4±5.9 | 72.8±5.2 | 76.0±7.5 | +3.2 |

For TwoRoom, Reacher, Push-T, and Cube, matched deltas compare against LeWM-cont.  For Wall, no continuation control is available, so the matched delta compares against local LeWM.

## Citation

```bibtex
@article{li2026predictive,
  title={Predictive but Not Plannable: RC-aux for Latent World Models},
  author={Li, Wenyuan and Li, Guang and Maeda, Keisuke and Ogawa, Takahiro and Haseyama, Miki},
  journal={arXiv preprint arXiv:2605.07278},
  year={2026}
}
```

This project builds on LeWorldModel, stable-worldmodel, stable-pretraining, DINO-WM, and LIBERO.  Please cite the corresponding original work when using those components.
