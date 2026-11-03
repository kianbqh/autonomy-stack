# FetchPush behavior cloning and domain randomization

I use a scripted pushing controller to study how behavior cloning changes when
demonstrations include randomized physics and sensing. The task uses the official
MuJoCo FetchPush environment with state observations and the native sparse reward.

## Measured results

Each trained policy was evaluated on the same 100 held-out resets. Both methods
used three training seeds and completed 200 epochs per seed.

| Method | Successes by seed, each /100 | Mean success | Population std, percentage points |
|---|---|---:|---:|
| BC | 5 / 4 / 7 | 5.33% | 1.25 |
| DR-BC | 4 / 9 / 22 | 11.67% | 7.59 |

DR-BC had higher mean nominal success, with substantial variation between seeds.
The [measured BC table](results/learning-20261103/comparison.md),
[six raw policy results](results/learning-20261103/raw/),
[collection counts and splits](results/learning-20261103/collection-summary.json),
and [fixed robustness panel](results/learning-20261103/panel-manifest.json)
retain the underlying measurements. The world-model module contains the paired
learned-MPC comparison.

All fits and evaluations used one CPU numerical thread. The environment receipt
records the installed CUDA-capable PyTorch build; these models were trained on CPU.

## Experiment

The comparison uses 100 successful demonstrations, nominal BC and DR-BC,
and three independent learning seeds 0/1/2. Both conditions attempt reset seeds
0..9999 in the same fixed order, independently retaining their first 100 native
step 50 successes and all failed attempts. Their successful pools can differ;
this comparison includes the change in data collection distribution. The same
reset-group hash 80/10/10 split determines fit, validation and unused test episodes.
Actual counts and expert collection success rates are disclosed in each manifest
and result. The pool budget is not the fit count. Both policies use the same
28→256→256→4 MLP, 200 epochs, Adam with learning rate 0.001, and batch size 256 and validation-action-MSE selection.
No held-out success outcome selects epochs, randomization settings or cells.

Final nominal outcomes use 100 fixed held-out resets 1000000..1000099 per trained
seed. Mean ± population std (`ddof=0`) is over the 3 policy rates, with per-policy
counts/Wilson95 intervals. Success is native `is_success` at control step 50;
ever-success and first-success remain separate diagnostics.

`configs/dr_eval.yaml` declares an 11-cell robustness panel: nominal and one-axis
low/high mass, friction, object-pose noise, command latency and Cartesian gain.
Each cell reuses the same 100 resets per seed. Panel means weight these 11 cells
equally; they do not represent the full Cartesian product or all conditions.

| Model | Parameters | Where applied |
|---|---|---|
| Mass |0.5/1/2 ×|Object mass and inertia together; restored before each reset|
| Friction |0.5/1/1.5 ×|Object and table geom friction; restored before each reset|
| Gaussian pose noise |sigma 0/5/10 mm|Visible object position, relative position and achieved_goal consistently; clean reward stays private|
| Dropout implementation |Fixed hold and Bernoulli trigger in config|Holds the same redundant visible pose fields; engineering checks only in main scope|
| Command latency |0/1/2 control steps|FIFO initialized with zero commands|
| Cartesian gain |0.8/1/1.2 ×|Mocap command delta before clipping; not seven-joint actuator gain|

The existing shield implementation filters commands at the actuation boundary,
after delay/gain, with workspace, speed and swept keep-out checks. Geometry tests
and physical audit support remain. A full fault/shield matrix is outside the main
experiment. FetchPush locks its gripper, so gripper failure is inert/N/A here.

From the repository root, create an isolated environment with the measured
Linux CPython 3.12 CPU dependency lock, then choose an explicit output directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-m3-ci.lock
export PYTHONPATH="$PWD/manipulation:$PWD/world_model"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export M3_RUN_ROOT="$PWD/runs/manipulation"
export INPUTS="$PWD/inputs.json"
python manipulation/scripts/runtime_smoke.py --config manipulation/configs/push_smoke.yaml --repo "$PWD" --run-root "$M3_RUN_ROOT"
bash manipulation/scripts/run_all.sh --stage collect --repo "$PWD" --run-root "$M3_RUN_ROOT" --inputs "$INPUTS"
bash manipulation/scripts/run_all.sh --stage bc --repo "$PWD" --run-root "$M3_RUN_ROOT" --inputs "$INPUTS"
bash manipulation/scripts/run_all.sh --stage panel --repo "$PWD" --run-root "$M3_RUN_ROOT" --inputs "$INPUTS"
bash manipulation/scripts/run_all.sh --stage report --repo "$PWD" --run-root "$M3_RUN_ROOT" --inputs "$INPUTS"
```

`INPUTS` is an actual-artifact JSON: `demo_manifests.nominal/dr`, a hashed
`policy_manifest` with real nominal/DR checkpoints for seeds 0/1/2, and
`report_inputs` with the six real BC results and panel summary. Missing inputs
fail explicitly. There is no automatic dependency install, download or schedule.
Windows can run these commands through WSL2. Keep large NPZ/checkpoints outside Git.
Each attempt records command/config/source/checkpoint hashes, environment, seeds,
timestamps, wall times and failures without overwriting old results.
Environment receipts detect the actual platform, WSL, runner and container
evidence. Optional `M3_ENVIRONMENT` is recorded as a user label alongside the
detected environment. An existing CUDA-enabled Torch 2.15 environment can be
selected with `M3_PYTHON`; its actual package/hardware identities are recorded.

Engineering checks cover the installed observation/timebase/XML contract,
native reward and timeout, scripted-controller floor, actual BC gradients,
complete clone restoration, parameter restoration, noise consistency, FIFO
delay/reset behavior and physical Cartesian gain. Gaussian noise distribution
calibration was not estimated.

## Limitations

This is state-based simulation, with a scripted demonstration source. It does not
measure real-robot or sim-to-real transfer. Sensor noise/delay causes partial observability. The shield
has no unconditional physical safety guarantee. Existing 2-D, safety,
perception and viewer measurements remain separate and frozen.

Environment definitions follow the official [FetchPush documentation](https://robotics.farama.org/v1.4.1/envs/fetch/push/)
and [MuJoCo bindings](https://mujoco.readthedocs.io/en/3.3.7/python.html).

## Evidence files

[Evidence index and hashes](results/learning-20261103/evidence-index.json) list the actual raw artifacts. Large repeated raw JSON is stored losslessly as `.json.gz`; `gzip -dc file.json.gz` recovers the exact source bytes. Derived tables are labelled and link to their raw inputs. Checkpoints and NPZ episodes stay outside Git. [Download the final models](https://github.com/kianbqh/autonomy-stack/releases/tag/m3-models-20261103); verify their [checkpoint hashes](../world_model/results/learning-20261103/checkpoint-downloads.json).
