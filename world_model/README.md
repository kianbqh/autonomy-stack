# Small learned world model and MPC

I train a small state encoder and action-conditioned latent dynamics model, then
use its physical decoder inside a bounded CEM controller. The experiment asks
whether recorded-action prediction errors explain closed-loop pushing success
relative to behavior cloning. MPC uses the known object-goal distance cost.

Actual model training, open-loop
errors, MPC success and latency are **not run**. A latent-model claim requires the
encoder, dynamics and decoder to receive actual training and reported evidence.

The single main configuration is 18→64→8 encoder, `(latent8,action4)`→64→8
residual latent dynamics, and 8→64→18 physical decoder, with SiLU hidden layers.
The objective combines next-state normalized MSE, 0.25 × current/next reconstruction
MSE and 0.1 × latent consistency. Normalizers use fit transitions and their supervised
labels only. Validation total loss selects a checkpoint; test data never selects
settings. Per-component gradients, reconstruction loss and latent dimension std
are retained to diagnose failed/collapsed learning. Reconstruction does not prove
a mathematical guarantee against collapse.

Fresh expert, frozen nominalBC seed 0 and Gaussian-noisyBC seed 0 each contribute
200 complete 50-step episodes, including failures. Seed 0 is declared before any
test result. Reset allocations are 800000..800199, 801000..801199 and 802000..802199;
held-out reset seeds never enter intake. The common reset-group SHA256 split is
80/10/10 across all sources and all 3 independently trained model seeds 0/1/2, even
if source allocations do not intersect. Actual source/split counts and IDs are
audited; no SAC replay or fabricated replacement is required by this experiment.

Episode intake verifies TrackA NPZ/sidecar and producer source/config/controller/
checkpoint hashes, installed simulator versions, XML includes, original 50-step
timeout/reward, exact observation fields and full integration-state arrays.
The 18 physical fields are grip position, object position, wrapped object Euler,
object world velocity, object angular velocity and grip velocity. Fetch native
velocities already include control_dt; relative object velocity is converted to
world units explicitly. Visible state and available policy command are model
inputs; clean state is supervision. Hidden queues and simulator state do not
enter ordinary MPC.

Open-loop scoring uses recorded actions at k = 1/5/10/20/30, encodes once, and rolls
the latent state without teacher forcing. Object/gripper errors are in mm, with
raw arrays and the constant-velocity counterpart. Mean/std first aggregate valid
starts inside each episode, then episode means; a separate 3-model-seed summary
uses population std. This single-model design makes no ensemble uncertainty claim.

The fixed main CEM uses horizon 10, 128 candidates, 3 iterations and 16 elites.
Actions are bounded [-1,1]; latent rollout is decoded for object-goal distance
plus 0.001 × squared-action cost. Effective horizon=min(10,50-policy_step); warm-start
resets every episode. Nominal success uses the same 100 held-out resets as BC,
for each of 3 model seeds. Synchronized GPU planner/control timing keeps raw
samples, 5 warmup steps on reset 910000, concurrency 1 and p50/p95 timer scopes.
No large sweep, five-member ensemble, full oracle or 243-cell MPC grid is required.

Use the isolated environment installed from the repository's
`requirements-m3-ci.lock`, and set actual input paths before running a stage:

```bash
export PYTHONPATH="$PWD/world_model:$PWD/manipulation"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OUT="$PWD/runs/world-model"
python world_model/scripts/collect_dataset.py --config "$COLLECTION" --output-root "$OUT"
python world_model/scripts/train_model.py --config world_model/configs/small_model.json --intake "$INTAKE" --output-root "$OUT"
python world_model/scripts/eval_prediction.py --config world_model/configs/prediction.json --intake "$INTAKE" --checkpoints "$CHECKPOINTS" --output-root "$OUT"
python world_model/scripts/eval_mpc.py --config world_model/configs/mpc.json --checkpoints "$CHECKPOINTS" --output-root "$OUT"
bash world_model/scripts/run_all.sh --config "$FROZEN_PLAN" --output-root "$OUT"
```

Copy the collection template and fill both BC checkpoint receipts with the same
actual nominalBC seed 0 artifact. Intake references the actual immutable manifest.
Freeze a config/hash dependency plan for collect/train/prediction/mpc/report/plot;
the empty template is rejected before launching a stage. `report` accepts actual
paired BC/DR-BC/MPC outputs, all 3 seeds each. Every stage is finite; nothing creates
a service or recurring task. Windows can use WSL2. Select explicit output paths
and keep large datasets/checkpoints outside Git. Result receipts include command,
Git/config/source/input identities, hardware, packages, timestamps and failures.
Platform, WSL, runner and container evidence determine the default environment
label; optional `M3_ENVIRONMENT` remains a user label, recorded separately from
the detected environment.

## Limitations

This is simulated state-based pushing with a known task cost and a scripted data
source, not a real robot or sim-to-real experiment. The small latent bottleneck
may discard contact/velocity information; hidden latency history is not modeled.
The fixed panel does not guarantee general robustness, and MPC can underperform
BC. Unit and engineering fixtures do not establish formal training or MPC results.
