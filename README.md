# Autonomy stack

[![Autonomy stack](https://github.com/kianbqh/autonomy-stack/actions/workflows/tests.yml/badge.svg)](https://github.com/kianbqh/autonomy-stack/actions/workflows/tests.yml)

I use this project to follow estimation error through mapping and control: fuse noisy sensors, register recorded LiDAR scans, learn pushing policies, and replay where the resulting trajectories diverge. A separate shelf-perception experiment trains and exports a product detector on SKU-110K.

Goal-reaching and obstacle safety use **2-D synthetic worlds**. The LiDAR module processes **recorded 3-D KITTI data**, while pushing uses **3-D MuJoCo FetchPush simulation**. Each has its own data, controls and evaluation. There is no physical robot or vehicle deployment. Retail perception remains an independent offline experiment.

```mermaid
flowchart TB
  subgraph planar[2-D localization and safety]
    sim[Unicycle / obstacle worlds] --> sensors[IMU + wheel speed + GPS]
    sensors --> ekf[Shared EKF]
    ekf --> control[Goal / avoidance policies]
    control --> sim
    sim --> trace[State and fault traces]
  end
  subgraph lidar[Recorded 3-D LiDAR]
    kitti[KITTI scans + calibration] --> icp[ICP odometry]
    icp --> graph[GTSAM loop graph]
    graph --> poses[Camera0 SE3 tracks]
    poses --> evaluation[evo / KITTI evaluation]
    gt[Ground truth] --> evaluation
  end
  trace --> viz[Three.js replay viewer]
  poses --> viz
  subgraph pushing[3-D MuJoCo FetchPush]
    fetch[FetchPush simulator] --> episodes[Expert and policy episodes]
    episodes --> cloning[BC / DR-BC]
    episodes --> model[Latent dynamics + CEM MPC]
    cloning --> fetch
    model --> fetch
  end
  subgraph perception[Offline retail perception]
    sku[SKU-110K photographs] --> yolo[YOLO11s / ONNX]
    yolo --> gaps[Geometric shelf-gap candidates]
  end
```

## Open the integrated replay

Use Python 3.12 and Node 24.15 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt pandas pillow pyarrow
bash scripts/run_stack.sh
```

Open `http://127.0.0.1:5176`. The command reruns two recorded safety cases, checks them against the original matrix, exports their state/fault traces and starts the viewer. Choose the successful normal-GPS episode or the GPS-bias collision. Playback, scrubbing, layer controls and the terminal fault/clearance display all use those exports. The 120-second fusion drive is also available.

![The normal safety scene with 100,000 procedural points](viz/results/native-gpu/screenshot_100k.png)

## Measured results

| Component | Result | Evidence |
|---|---|---|
| Fusion, 50 × 120 s drives | Gated EKF position RMSE 0.979 m, held GPS 2.660 m: 63.2% lower | [Fusion metrics](results/fusion_metrics.json) |
| Learned controller, 200 held-out goals | 197/200 reached with EKF observations; 165/200 with raw observations | [Policy metrics](results/policy_metrics.json) |
| Actual ROS 2 loop | 180 s launch, 175.05 s recorded metrics; fused RMSE 0.905 m vs held GPS 2.894 m | [ROS metrics](results/ros2_metrics.json) |
| Safety matrix, 3,000 simulated episodes | Gated EKF 178/600 collisions; uncertainty monitor 61/600, with more timeouts | [Safety report](safety/results/safety_report.md) |
| Native Windows RTX 3080 rendering | 10k / 100k / 1M points: 535.30 / 572.39 / 433.20 FPS | [Every frame interval and renderer](viz/results/native-gpu/bench.json) |
| Full SKU-110K detector | YOLO11s, 100 epochs; official 588-image validation mAP50 0.9281 / mAP50–95 0.5782 | [Training and validation](perception/results/train_full_3080.json) |
| Detector ONNX export, CPU raw forward | 52.5 ms p50 / 53.9 ms p95, batch 1, four threads; raw output parity passed | [All timing samples and model hashes](perception/results/onnx_bench_full.json) |

Fusion, control, ROS and the full safety matrix were measured locally on 31 October 2026. The integration reruns the two exported safety cases and verifies their outcomes, duration, minimum clearance and localization RMSE against those saved matrix rows. [Replay checks and hashes](results/replay_integration.json).

The hardware rendering run used native Windows Chromium 156, NVIDIA driver 621.01 and actual RTX 3080 / ANGLE D3D11. It measured three-second windows at 1280×720 after 31 warmup intervals, synchronizing each GPU frame with `readPixels`. The safety scene has 12 draw calls. These are not a single-factor speedup comparison with the older fusion-only SwiftShader scene. [Rendering protocol and interactions](viz/README.md).

The detector used the full official training split: 8,216 usable images and 1,207,996 product boxes after loader verification. Its reported scores come from a separate validation of the selected best checkpoint on all 588 validation images. [Detector setup, artifacts and CPU protocol](perception/README.md).

[Download the trained weights and ONNX package](https://github.com/kianbqh/autonomy-stack/releases/download/sku110k-yolo11s-v1/yolo11s-sku110k-full.zip). The package includes the original model bytes, attribution and manifest; [download checksum](perception/results/model_release.json) and [individual model hashes](perception/results/model_artifacts.json) identify them.

## Recorded SLAM and 3-D pushing

The LiDAR front end, pose graph, metric adapters and FetchPush experiments are separate from the measured 2-D runs above. The [six CI jobs at `75bb06e`](https://github.com/kianbqh/autonomy-stack/actions/runs/41585516271) passed, including 15 native C++ tests, actual synthetic positive/negative SLAM fixtures, evaluator checks and learning CPU contracts.

| Experiment | Measured result | Evidence |
|---|---|---|
| KITTI sequence 07 | Complete 1,101-frame tracks; ATE 0.563064 → 0.370273 m after the loop graph; 2 accepted loops | [Full metrics and original trajectories](slam/results/kitti07-20261103/REAL_METRICS_SUMMARY.json) |
| Distance-pair error | 100 m translation RPE worsens from 0.765796 to 0.790351 m | [Independent evo/devkit evaluation](slam/results/kitti07-20261103/evaluation-corrected-75bb/metrics.json) |
| Camera–LiDAR diagnostic | Three recorded frames; supplied-transform mean score 0.57336; best sampled tz offset +0.025 m | [Scores and support](slam/results/kitti07-20261103/calibration-original51a4/calibration.json) |
| Recorded 3-D replay | Full1101, original real27 failure and synthetic2 failure pass; complete route fits desktop and narrow views | [Browser receipts and screenshots](slam/results/kitti07-20261103/browser/README.md) |
| FetchPush BC / DR-BC | Nominal held-out success: BC 5% / 4% / 7%, DR-BC 4% / 9% / 22%, seeds 0 / 1 / 2 and 100 resets each | [Pushing comparison](manipulation/README.md) |
| Learned object rollout | One-step error 23.401 mm vs constant velocity 3.446 mm; 30-step error 126.085 vs 284.348 mm | [Prediction table](world_model/results/learning-20261103/prediction-table.md) |
| FetchPush MPC | Mean nominal success 6% vs DR-BC 11.67%; shared-host CPU planner p50/p95 3.424/4.375 ms | [Control outcomes](world_model/results/learning-20261103/measured-success-table.md), [timing scope](world_model/results/learning-20261103/cpu-timing-table.md) |

The loop graph improves global position ATE while the 100 m translation RPE increases. Both outcomes are retained. The completed run uses uniform stopping precision of 1 mm translation and 0.0001 rad rotation, with the original 40-iteration budget and quality gates. The [SLAM guide](slam/README.md) records this engineering change, earlier failures and the distinct native/evaluator identities.

Calibration retains the original, identical calibration fields. Five axis sweeps favor zero offset; the three-frame tz optimum is a sensor-consistency observation, and the supplied transform was left unchanged. The six BC fits used identical manipulation source bytes across `c9254b2` and `61d485e`; their low success rates and seed variation remain part of the result. The learned model has larger one-step object error than constant velocity and smaller 30-step error; MPC success is below DR-BC in this measured comparison. The [FetchPush guide](manipulation/README.md) and [world-model experiment](world_model/README.md) describe the data, fixed robustness panel and latent-model/CEM comparison.

## Modules and data

- [`fusion_core/` and ROS 2](docs/fusion.md): noisy synthetic IMU, odometry and GPS; EKF, learned policy, Python experiments and the existing 180-second ROS runner. `bash scripts/run_ros2_docker.sh 180` runs the separate ROS nodes in the official Humble image.
- [`safety/`](safety/README.md): 30 fault/noise cells, five estimators and 20 seeds, with known circular obstacles. Exports normal and collision cases using the shared simulator/filter rather than a copied implementation.
- [`slam/`](slam/README.md): C++ LiDAR registration, descriptor-based loop proposals, GTSAM graph optimization, KITTI/evo evaluation and a camera–LiDAR consistency diagnostic.
- [`manipulation/`](manipulation/README.md): state-based MuJoCo FetchPush, scripted demonstrations, nominal BC and domain-randomized BC.
- [`world_model/`](world_model/README.md): learned latent dynamics, recorded-action prediction and CEM model-predictive pushing control.
- [`viz/`](viz/README.md): safety truth/estimate/covariance/fault inspection, a separate recorded-SLAM entry and a procedural point-cloud stress scene. The stress cloud is synthetic; recorded SLAM replay supplies trajectory data.
- [`perception/`](perception/README.md): official retail photographs and product-box labels. Its box-spacing heuristic produces gap candidates; the dataset does not label gaps or robot obstacles.

## Checks

```bash
PYTHONPATH=.:safety:perception python -m pytest --capture=sys -q tests safety/tests perception/tests
(cd viz && npm ci && npm run typecheck && npm test && npm run build)
```

One workflow runs six jobs: simulation, ROS 2, viewer/browser correctness, SLAM Python, native SLAM CPU and learning CPU contracts. The latter modules use their own pinned dependency locks and reproduction instructions. CI browser rendering is explicitly labeled SwiftShader software; the hardware table comes from the retained local NVIDIA run. Large datasets, training checkpoints and dependency caches stay outside Git.

The source repositories were merged with unsquashed subtrees. Their original commits remain ancestors; [history provenance](results/source_history.json) records the source HEADs, actual import commands and ancestry checks.
