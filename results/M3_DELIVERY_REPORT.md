# M3 delivery report: learning control and offline LiDAR SLAM

I compared behavior cloning, domain-randomized demonstrations and a small learned-model controller in FetchPush, then built and evaluated an offline LiDAR SLAM pipeline on KITTI sequence 07. This report records the completed reduced experiment and the original negative outcomes.

## Publication and source identities

- Main merge: [43f4823](https://github.com/kianbqh/autonomy-stack/commit/43f4823634835978fa2ca321d25d02f394850fdf) via [PR #1](https://github.com/kianbqh/autonomy-stack/pull/1).
- Validated publication head: [30b7630](https://github.com/kianbqh/autonomy-stack/commit/30b7630c2859f4618c1b1f9160b818c2dc668847); all six jobs passed in [CI run 41587571238](https://github.com/kianbqh/autonomy-stack/actions/runs/41587571238). The merge receipt records an identical tree.
- All six jobs also passed on merged main in [CI run 41588332476](https://github.com/kianbqh/autonomy-stack/actions/runs/41588332476).
- [Download the nine final models](https://github.com/kianbqh/autonomy-stack/releases/tag/m3-models-20261103). Individual asset URLs and SHA256 values are listed below.
- [Machine-checked metrics/source map](M3_DELIVERY_METRICS.json) retains full-precision values, input hashes and check counts. Tables display measured values; percentages are converted from the saved rates.

Collection and the first two BC fits used `c9254b250e8e645a7574c7bfa1d1137d66dd9613`. The remaining four fits and the world-model chain used `61d485e79d8f1cd4ef5791f148c853555f4db390`; manipulation source bytes were identical across all six fits. Native SLAM used the same `61d485e` generation. The corrected evaluator used `75bb06e7b114a2f004d0d8c44348864e728095f8`. The viewer checks retain their separate `a5139856` identity.

## Execution environment

| Component | Recorded environment |
|---|---|
| Host/runtime | Windows host; Linux x86_64 WSL2, kernel 6.18.54.1-microsoft-standard-WSL2, glibc 2.39 |
| CPU | Intel Core i7-12700KF; native receipt records 20 logical CPUs |
| GPU / driver | NVIDIA GeForce RTX 3080, 12,288 MiB, driver 621.01; formal learning and SLAM used CPU |
| Python / numerical stack | CPython 3.12.3; NumPy 2.2.6; one numerical thread for learning/MPC |
| Simulator | MuJoCo 3.3.7; Gymnasium 1.2.0; Gymnasium-Robotics 1.4.1 |
| Torch / CUDA | Imported Torch 2.15.0+cu132 (CUDA 13.2 build tag); formal tensors on CPU with CUDA hidden |
| Portable CPU CI | Separate hash-locked Torch 2.15.0+cpu environment; CPU identity assertion passed in final CI |
| Stable-Baselines3 | Not installed in the formal learning environment; SAC/PPO were outside the reduced scope |
| Native build | GCC 13.3; CMake 4.4.4; Ninja 1.13.2; Boost 1.83; Eigen 3.4.0; nanoflann 1.6.3; GTSAM 4.2 |
| Evaluator | evo 1.31.0; SciPy 1.15.3; OpenCV-headless 4.11.0.86; matplotlib 3.10.3 |

The local experiment used isolated environments on the G-backed WSL disk and reused the shared Torch distribution read-only. The separate portable CPU lock supports CI reproduction. Formal experiment timing was collected on a shared host with ordinary user workloads present. A bounded same-shape engineering probe selected CPU; it was not an exclusive hardware benchmark. Formal runs did not use Docker. The ROS 2 CI job is a separate retained pipeline.

[Learning environment and device receipts](../world_model/results/learning-20261103/provenance/); [SLAM build receipt](../slam/results/kitti07-20261103/native-run/build-receipt.json); [evaluation lock](../slam/evaluation-environment.lock.json); [CPU dependency lock](../requirements-m3-ci.lock).

## Track A — behavior cloning and domain randomization

Each condition collected its first 100 final-step successes in reset order starting at 0, retaining every failed attempt. Episodes contain 50 controls. A common reset-group SHA256 rule assigned the 80/10/10 split; successful pools differed between conditions.

| Condition | Attempts | Successes | Failures retained | Train / val / unused test episodes | Collection wall s |
|---|---|---|---|---|---|
| nominal | 362 | 100 | 262 | 78/14/8 | 37.40728249400854 |
| dr | 325 | 100 | 225 | 81/10/9 | 28.387073896999937 |

Collection wall time is stage accounting; the nominal attempt also included the source-defined 100-episode expert evaluation. Engineering samples were not reused.

The six fixed fits used a 28→256→256→4 MLP, Adam 0.001, batch size 256 and 200 epochs. Minimum grouped validation action MSE selected each checkpoint. Every final evaluation used the same 100 resets, 1000000..1000099, and native `is_success` at control step 50.

| Method | Seed | Fit / val / unused-test episodes | Epochs / selected | Min validation action MSE | Nominal successes /100 |
|---|---|---|---|---|---|
| BC | 0 | 78/14/8 | 200/145 | 0.05573846399784088 | 5 |
| BC | 1 | 78/14/8 | 200/184 | 0.050240468233823776 | 4 |
| BC | 2 | 78/14/8 | 200/189 | 0.05169672146439552 | 7 |
| DR-BC | 0 | 81/10/9 | 200/98 | 0.13312339782714844 | 4 |
| DR-BC | 1 | 81/10/9 | 200/198 | 0.12109298259019852 | 9 |
| DR-BC | 2 | 81/10/9 | 200/193 | 0.12538979947566986 | 22 |

| Method | Nominal mean % | Population std, percentage points | 11-cell mean % | Population std, percentage points |
|---|---|---|---|---|
| BC | 5.333333333333334 | 1.2472191289246475 | 5.272727272727273 | 1.4142135623730954 |
| DR-BC | 11.666666666666666 | 7.586537784494028 | 12.181818181818183 | 5.991729837557848 |

The fixed panel has 11 equally weighted one-axis cells, six policies and 6,600 terminal outcomes. Each triplet below gives successes for seeds 0/1/2, each out of 100. This is a panel average, not the full parameter-product grid.

| Cell | BC successes by seed | BC mean % | DR-BC successes by seed | DR-BC mean % |
|---|---|---|---|---|
| nominal | 5/4/7 | 5.333333333333334 | 4/9/22 | 11.666666666666666 |
| mass_low | 5/6/9 | 6.666666666666667 | 4/3/16 | 7.666666666666668 |
| mass_high | 3/3/12 | 6.0 | 6/10/19 | 11.666666666666666 |
| friction_low | 1/6/7 | 4.666666666666667 | 8/16/21 | 15.0 |
| friction_high | 6/5/5 | 5.333333333333334 | 8/9/23 | 13.333333333333334 |
| noise_sigma_low | 3/4/7 | 4.666666666666667 | 6/6/20 | 10.666666666666668 |
| noise_sigma_high | 4/2/4 | 3.3333333333333335 | 7/10/31 | 16.0 |
| latency_low | 8/5/12 | 8.333333333333332 | 6/15/20 | 13.666666666666668 |
| latency_high | 7/5/4 | 5.333333333333334 | 9/11/14 | 11.333333333333334 |
| gain_low | 2/4/10 | 5.333333333333334 | 5/14/18 | 12.333333333333334 |
| gain_high | 3/3/3 | 3.0 | 5/7/20 | 10.666666666666668 |

Mass factors were 0.5/2, friction 0.5/1.5, pose-noise sigma 5/10 mm, FIFO delay 1/2 controls and Cartesian mocap gain 0.8/1.2, with other fields nominal. Physical/sensor engineering checks covered actual Fetch stepping, clone restoration, timeout/reward, gradient flow, FIFO/reset behavior and shield command paths. The full fault-by-shield study was not part of the main experiment.

[Collection/splits](../manipulation/results/learning-20261103/collection-summary.json); [all panel rows](../manipulation/results/learning-20261103/robustness-panel.csv); [training history CSV](../manipulation/results/learning-20261103/bc-training-history.csv); [raw results and hashes](../manipulation/results/learning-20261103/evidence-index.json).

## Track B — small world model and MPC

The single main architecture is an 18→64→8 encoder, residual `(latent 8, action 4)`→64→8 dynamics and 8→64→18 decoder, with SiLU hidden layers. The loss combines normalized next-state MSE, 0.25 × reconstruction and 0.1 × latent consistency. Expert, predeclared nominal-BC seed 0 and noisy-BC seed 0 each supplied 200 complete episodes, including failures. The BC producer was fixed before test outcomes.

| Source | Train episodes / transitions | Validation episodes / transitions | Test episodes / transitions |
|---|---|---|---|
| bc | 159 / 7950 | 17 / 850 | 24 / 1200 |
| expert | 166 / 8300 | 13 / 650 | 21 / 1050 |
| noisy_bc | 168 / 8400 | 16 / 800 | 16 / 800 |

| Model seed | Epochs completed | Selected epoch | Best validation total loss |
|---|---|---|---|
| 0 | 100 | 98 | 0.3166725622571033 |
| 1 | 100 | 94 | 0.312080565172693 |
| 2 | 67 | 57 | 0.31501624838165615 |

Recorded-action prediction encodes once and rolls without teacher forcing. Valid starts are averaged within each episode, then across 61 held-out episodes and the three independently trained model seeds. The constant-velocity reference is shared rather than trained three times.

| Horizon | Object learned mm | Object CV mm | Gripper learned mm | Gripper CV mm |
|---|---|---|---|---|
| 1 | 23.401472576328967 | 3.4461887989106503 | 26.97135087831424 | 11.635141880948398 |
| 5 | 38.00358521091482 | 38.854238557288525 | 37.78637763580989 | 81.63481665329424 |
| 10 | 57.76727784787371 | 100.66661119573989 | 56.70831534380469 | 181.20086568045875 |
| 20 | 93.03082209727802 | 211.7233542704415 | 97.5250969037332 | 377.14957726687607 |
| 30 | 126.08459687363255 | 284.34767612101626 | 126.89802229209026 | 562.9601693750093 |

There are 915 saved episode-horizon records (3 × 61 × 5). [Prediction plot, seed 0](../world_model/results/learning-20261103/figures/prediction-seed-0.png), [seed 1](../world_model/results/learning-20261103/figures/prediction-seed-1.png) and [seed 2](../world_model/results/learning-20261103/figures/prediction-seed-2.png) show population-standard-deviation error bars across the 61 episodes for that model seed, not across model seeds.

MPC uses CEM horizon 10, 128 candidates, three iterations and 16 elites, bounded actions and decoded object-goal distance plus 0.001 × squared action cost. The effective horizon is limited by the remaining controls, and warm-start state resets each episode.

| Method | Successes by seed, each /100 | Mean success % | Population std, percentage points |
|---|---|---|---|
| BC | 5/4/7 | 5.333333333333334 | 1.2472191289246475 |
| DR-BC | 4/9/22 | 11.666666666666666 | 7.586537784494028 |
| Learned MPC | 6/5/7 | 6.0 | 0.8164965809277263 |

Timing retains 15,000 measured controls after five warmup controls per model seed on reset 910000, concurrency 1. Resets, pre-act deepcopy, episode summarization and disk I/O are excluded. Full-control time runs from policy entry through `env.step` return, including preprocessing, planning, host copies and native physics/sensor wrappers.

| Shared-host CPU scope | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|
| planner_ms | 3.341671061498346 | 3.4241905086673796 | 4.375198460184038 | 5.095899013685994 |
| host_policy_ms | 3.4091480468322213 | 3.489590497338213 | 4.461116562015376 | 5.200384480704088 |
| policy_act_ms | 3.4832339639775456 | 3.5641804861370474 | 4.5472288693417795 | 5.294945526111405 |
| full_control_step_ms | 4.555765604470313 | 4.631354502635077 | 5.730770390073303 | 6.552856234484352 |

[Raw timing CSV](../world_model/results/learning-20261103/raw-timings.csv); [paired terminal outcomes](../world_model/results/learning-20261103/paired-policy-outcomes.csv); [raw WM results](../world_model/results/learning-20261103/evidence-index.json); [original and derived CPU plans](../world_model/results/learning-20261103/configs/).

## Track C — offline KITTI LiDAR SLAM

Sequence 07 completed all 1,101 original frames/timestamps with valid SE(3) odometry and graph tracks, 381 keyframes and two accepted loop factors. Range/voxel filtering, point-to-plane ICP, Scan-Context-style proposals, reciprocal ICP verification and an end-of-sequence GTSAM graph form the pipeline. GT enters post-hoc evaluation, not registration or loop decisions.

| Metric | ICP odometry | Loop graph | Unit |
|---|---|---|---|
| ATE (SE3, no scale) | 0.5630644298531414 | 0.370272684080678 | m |
| RPE one frame, translation | 0.017406744210859974 | 0.017402571911432536 | m |
| RPE one frame, rotation | 0.06093751678146186 | 0.06092707636065464 | deg |
| RPE 100m, translation | 0.7657957862712936 | 0.7903507992059837 | m |
| RPE 100m, rotation | 0.7564551961131829 | 0.7487282283936924 | deg |
| Official KITTI translation | 0.6881891284137964 | 0.6811798550188541 | % |
| Official KITTI rotation | 0.5102498572424977 | 0.508066856413105 | deg / 100m |

ATE uses no-scale SE(3) alignment. One-frame and 100 m RPE use original frame pairs; the 100 m result has 881 exact pairs. The separate official KITTI comparison has 317 valid segments. All ten saved evo error arrays and the official segment identities/errors passed independent comparison. Both tracks come from the same native pose bytes.

| Loop diagnostic | Count |
|---|---|
| proposed | 1854 |
| accepted | 2 |
| true_positives | 2 |
| false_positives | 0 |
| rejected_true_candidates | 18 |

Loop labels use GT distance strictly below 4 m after execution. The proposal log includes temporal exclusions and rejected candidates; these counts do not define recall over every possible revisit.

| Shared-host native scope | Samples | Mean ms | p95 ms |
|---|---|---|---|
| io_ms | 1100 | 3.524413186363636 | 4.40760785 |
| preprocess_ms | 1100 | 4.172274058181818 | 5.037271999999999 |
| map_normals_ms | 1100 | 13.855952353636363 | 50.76979735 |
| registration_ms | 1100 | 10.68975913909091 | 20.074814149999998 |
| loop_ms | 1100 | 1.009790410909091 | 0.23799069999999997 |
| frame_inclusive_ms | 1100 | 33.353458372727275 | 77.51549384999998 |
| frame_cpu_ms | 1100 | 33.35223727272727 | 77.51209999999999 |

Final graph optimization took 9.218664 ms. Native process wall time was 36.75623135801288 s and process CPU time was 36.743934 s. Samples cover 1,100 non-origin frames, with no warmup or repeated passes. Frame-inclusive time includes I/O, preprocessing, registration/map/loops and accepted-prefix output; final diagnostic flush is counted in process wall time.

The successful runtime config hash is `ee689925bcecd2ee7c71817e47903a2c677c7c8dd09684ef2c7e5067d9d6c5da`. Its stopping precision is 0.001 m translation / 0.0001 rad rotation, versus 0.0001 m / 0.00001 rad in the original config. Frame-114 geometry diagnostics exposed submillimeter correspondence cycling. The uniform precision change retained the 40-iteration budget and all fitness, inlier, conditioning and loop gates. Earlier frame-27, frame-50 and strict-frame-114 failures remain attached to their original attempts.

The camera–LiDAR diagnostic projected image_2 frames 0, 550 and 1100 with `P2 * Tr * p_L`, positive-depth/image masks and a nearest-depth z buffer. The supplied-transform composite score was 0.5733597079912821 (lower is better).

| Axis | Best sampled offset | Unit | Mean score at sampled best |
|---|---|---|---|
| rx | 0 | degrees | 0.5733597079912821 |
| ry | 0 | degrees | 0.5733597079912821 |
| rz | 0 | degrees | 0.5733597079912821 |
| tx | 0 | meters | 0.5733597079912821 |
| ty | 0 | meters | 0.5733597079912821 |
| tz | 0.025 | meters | 0.57168581088384 |

The +0.025 m sampled tz optimum was not applied to the calibration. Three frames and one-axis sweeps support an offline sensor-consistency diagnostic, not recovered extrinsics.

[Exact seven-row summary and timing samples](../slam/results/kitti07-20261103/REAL_METRICS_SUMMARY.json); [corrected evaluator output](../slam/results/kitti07-20261103/evaluation-corrected-75bb/metrics.json); [trajectory/loops figure](../slam/results/kitti07-20261103/evaluation-corrected-75bb/trajectory-loops.png); [calibration scores and support](../slam/results/kitti07-20261103/calibration-original51a4/calibration.json); [calibration sweep plot](../slam/results/kitti07-20261103/calibration-original51a4/score-shape.png); [frame 550 overlay](../slam/results/kitti07-20261103/calibration-original51a4/overlay-000550.png).

The recorded viewer checks covered full 1,101-frame replay, the original real frame-27 failed prefix and an explicitly synthetic degeneracy fixture. The focused desktop/narrow regression used headless SwiftShader and put all 3,303 track points inside the clip volume. [Browser evidence](../slam/results/kitti07-20261103/browser/README.md) retains captures, functional checks and separate source identities.

## Negative and surprising results

- DR-BC averaged 11.67% nominal success versus BC 5.33%, but individual DR seeds ranged from 4% to 22%.
- Learned MPC averaged 6% success, below DR-BC despite lower 30-step object prediction error (126.08459687363255 mm versus CV 284.34767612101626 mm).
- One-step object prediction was worse than constant velocity: 23.401472576328967 mm versus 3.4461887989106503 mm.
- Graph ATE improved from 0.5630644298531414 m to 0.370272684080678 m, while 100 m translation RPE worsened from 0.7657957862712936 m to 0.7903507992059837 m.
- Both accepted loops were post-hoc true positives, but 18 true candidates were rejected; accepted-loop precision does not establish revisit recall.
- Calibration favored a sampled +0.025 m tz offset; that observation was left as a diagnostic and did not alter input extrinsics.

## Scope cuts and remaining inputs

The reduced November 2 scope superseded the broader initial plan. The following were intentionally not run:

| Not run | Reason |
|---|---|
| Diffusion policies; SAC/HER, BC-to-SAC and PPO | Removed from the learning comparison; delivered methods are BC, DR-BC and one small learned-model MPC configuration |
| Demonstration scaling and steps-to-50%/80% studies | One fixed pool of 100 successful demonstrations per condition; there is no RL sample-efficiency curve |
| Full Cartesian randomization sweep and fault-by-shield on/off matrix | Replaced by the fixed 11-cell panel and bounded physical/sensor/shield engineering checks |
| World-model ensembles, parameter sweeps and full simulator-oracle MPC | One main configuration with three independent training seeds; no oracle or ensemble claim |
| KITTI sequence 00 | Optional extension not executed; this release reports complete sequence 07 only |
| Real-robot, sim-to-real and onboard real-time validation | The delivered control study is simulation and the SLAM study is offline recorded-data evaluation |

Needs Phot: no additional input is required for this reduced release. A future hardware experiment or new dataset would require its own acquisition plan and measurements. All finite learning processes are closed; no additional rollout suite was added after the fixed plan.

## Factual scope

The artifacts support behavior cloning, domain-randomized simulation, a small learned dynamics model with CEM control, point-to-plane ICP, verified loop factors, a GTSAM pose graph and offline KITTI evaluation. They do not establish real-robot deployment, sim-to-real transfer, recovered extrinsics, exhaustive robustness, ensemble uncertainty or exclusive-host/onboard timing. The sensor shield has no unconditional physical safety guarantee.

## Frozen prior results

Exactly 50 pre-existing result files from baseline `39889ab511b91e35107abb7fe9eb0e7735360c1c` remain byte-identical. The frozen audit `FROZEN_M3_M4_20261103.json`, dated 2026-11-03T06:12:21.362672+00:00, records `status=pass`, `mismatches=[]`, and `exact_old_path_diff=""` for autonomy-stack. The report assembly also rechecked the current byte lengths and SHA256 values of all 50 paths.

[Frozen paths, SHA256 values and audit identity](M3_FROZEN_BASELINE.json) provide the exact path set. The corresponding old-path diff-stat command is:

```bash
python - <<'PY'
import json, subprocess
with open("results/M3_FROZEN_BASELINE.json") as handle:
    paths = [row["path"] for row in json.load(handle)["paths"]]
subprocess.run(["git", "diff", "--stat",
                "39889ab511b91e35107abb7fe9eb0e7735360c1c", "--", *paths], check=True)
PY
```

Observed diff-stat stdout: `""` (zero bytes). New results use new paths.

## Final model downloads

The publisher downloaded all nine public assets and verified their SHA256 values against the actual final models. Per-epoch checkpoints and raw NPZ episodes stay outside Git.

| Asset | Bytes | SHA256 |
|---|---|---|
| [bc-seed-0.pt](https://github.com/kianbqh/autonomy-stack/releases/download/m3-models-20261103/bc-seed-0.pt) | 309589 | `ba236762612836210ffbcc40cb5819eda64caacc76a43aae86cd3acf70bed203` |
| [bc-seed-1.pt](https://github.com/kianbqh/autonomy-stack/releases/download/m3-models-20261103/bc-seed-1.pt) | 309589 | `ff19038e1f7ab003d355d5a17143e433c6455bc018fbf909f55a300498ca53ae` |
| [bc-seed-2.pt](https://github.com/kianbqh/autonomy-stack/releases/download/m3-models-20261103/bc-seed-2.pt) | 309589 | `82543d8431be6737f21794c0295c97b2646ee72286e2131778bcf11184c779f0` |
| [bc_dr-seed-0.pt](https://github.com/kianbqh/autonomy-stack/releases/download/m3-models-20261103/bc_dr-seed-0.pt) | 309589 | `63194a89ff0c492af24559fab79aa7e30d278ebca822e05213cf70aa58c99e59` |
| [bc_dr-seed-1.pt](https://github.com/kianbqh/autonomy-stack/releases/download/m3-models-20261103/bc_dr-seed-1.pt) | 309589 | `085ab76ee68280ebf373bfb7d3686519dfb99ffdeda01a38a3999ed4682ecb2d` |
| [bc_dr-seed-2.pt](https://github.com/kianbqh/autonomy-stack/releases/download/m3-models-20261103/bc_dr-seed-2.pt) | 309589 | `b26b67e62609086e3f56ab8b48abfdc7c64cfccc3fd6e38a8ebab89bd5ed055d` |
| [small-latent-seed-0.pt](https://github.com/kianbqh/autonomy-stack/releases/download/m3-models-20261103/small-latent-seed-0.pt) | 254451 | `b5466ed8a792c9e4e68a045f407d1f1b7fe2bb19958c2722d3c9cb69c6b0df09` |
| [small-latent-seed-1.pt](https://github.com/kianbqh/autonomy-stack/releases/download/m3-models-20261103/small-latent-seed-1.pt) | 254451 | `bad47523fa7e2b1f01b039406a49e6a282f1d6cd5da2662e27374fa86746005e` |
| [small-latent-seed-2.pt](https://github.com/kianbqh/autonomy-stack/releases/download/m3-models-20261103/small-latent-seed-2.pt) | 254451 | `de327f07cd1efd54ab4b384b5ccffa41620b265f52c9d73219e95930bda83f7e` |

[Published checkpoint reference file](../world_model/results/learning-20261103/checkpoint-downloads.json) retains the verified release status and asset IDs. The source-generated result files remain immutable; large raw JSON is stored losslessly as `.json.gz`.
