# Safety under sensor faults

I wanted to see how sensor faults turn localization error into missed clearance, and whether a small uncertainty-aware speed and margin rule helps. This project runs a differential-drive robot past two known circular obstacles, using the estimated pose to steer toward a goal 12–20 m away.

The runner imports the simulator and EKF directly from the repository's shared [`fusion_core`](../fusion_core). The original matrix used the same implementation at commit `d47c6e0`; [file hashes](results/fusion_core_provenance.json) retain that provenance. Obstacles, sensor faults and motion are synthetic 2-D simulation; no hardware or camera detections enter the controller.

## Experiment

| Axis | Values |
|---|---|
| GPS noise σ | 0.5, 1.5, 3.0 m |
| GPS fault | None; 10 s dropout; 30 s dropout; 15 m spikes on 20% of fixes; 4 m bias for 15 s |
| Odometry fault | Normal; 10 s dropout |

The 30 scenario cells each run 20 seeds with five estimators, giving **3,000 episodes**:

- `truth`: oracle pose, showing the controller's behavior with exact localization.
- `raw`: held GPS position, gyro-integrated heading and odometry speed.
- `ekf_nogate`: EKF without innovation gating.
- `ekf`: EKF with χ² innovation gating.
- `ekf_monitor`: gated EKF plus an uncertainty-aware monitor.

The monitor adds 3σ to the controller's 0.6 m obstacle margin, with σ taken from the largest eigenvalue of the EKF position covariance. It reduces the speed cap above 0.5 m uncertainty and stops above 3 m. These thresholds are fixed throughout the experiment.

The simulator runs at 100 Hz and the controller at 20 Hz. The robot radius is 0.3 m. An episode ends on a sampled collision, arrival at the goal, or a 90 s time limit.

## Local results

All **14 tests passed**, including the oracle's zero localization error, fault windows, repeatability, avoidance, monitor response, gating and Wilson bounds. The full matrix completed in **166.78 s with three workers** on an i7-12700KF / Ubuntu 24.04 WSL2. [Run record](results/run_metadata.json) · [environment](results/runtime_environment.json) · [pytest output](results/pytest.log).

Each row below contains 600 episodes. Rates weight episodes equally; the collision interval is the Wilson 95% interval using that episode count.

| Estimator | Collision rate [95% CI] | Success | Timeout | Median minimum clearance | p95 maximum localization error | Mean episode NEES exceedance fraction |
|---|---|---|---|---|---|---|
| Truth | 0.000 [0.000, 0.006] | 1.000 | 0.000 | 0.931 m | 0.000 m | — |
| Raw | 0.723 [0.686, 0.758] | 0.157 | 0.120 | −0.001 m | 17.111 m | — |
| EKF, no gate | 0.442 [0.402, 0.482] | 0.403 | 0.155 | 0.120 m | 9.532 m | 33.14% |
| EKF | 0.297 [0.262, 0.334] | 0.593 | 0.110 | 0.516 m | 9.309 m | 20.89% |
| EKF + monitor | 0.102 [0.080, 0.128] | 0.597 | 0.302 | 1.302 m | 8.842 m | 28.87% |

A few observations from this matrix:

- The monitor reduced collisions from **178/600 to 61/600**, a 65.7% relative reduction. Timeouts rose from 66/600 to 181/600, and median time to goal among successes rose from 29.75 s to 55.15 s.
- Across the GPS-spike cases, ungated and gated collision rates were 0.683 and 0.225. Their median localization RMSEs were 1.524 m and 0.652 m.
- Bias-step cases had the largest residual collision rate for the gated filter: 0.400, versus 0.225 with the monitor. These are observed outcomes; the experiment does not identify a gate-lockout mechanism.
- The monitor still had collisions in 19 of 30 scenario cells. Its clearance and availability tradeoff is visible in the full cell results.

[Full safety report](results/safety_report.md) · [all episodes](results/episodes.csv) · [150 cell/estimator summaries](results/cells.csv) · [estimator summaries](results/estimator_summary.csv).

## Metric details

Minimum clearance is sampled from the true robot and obstacle geometry, including both radii. Negative clearance means overlap at a simulation step. Localization error compares the **updated estimate and truth at the same time**, at each controller-update endpoint; all 600 oracle episodes have zero RMSE and zero maximum error.

Position NEES uses the 2-D position error and covariance. For each episode, the runner computes the fraction of sampled values above 5.99. The reported fraction is the **equal-weight average of those episode fractions**; NEES means are also averaged per episode.

A fault regime was active at closest approach in 68.9% of faulted episodes. This uses the injector's time-window definition; the spike-enabled regime counts as active throughout its episode, while individual GPS fixes are spiked with probability 0.2.

These results describe the kinematic simulation, known obstacles and fixed controller used here. [Matrix and result checks](results/validation.json).

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
PYTHON="$VIRTUAL_ENV/bin/python" bash scripts/run_all.sh
```

Run these commands from `safety/`. `SEEDS=20` and `WORKERS=3` are the defaults. Each scenario is checkpointed to `results/episodes.csv`; `RESUME=1` reuses saved scenario rows. The requested scenario/estimator/seed coverage is checked before aggregation. The root workflow runs these 14 tests alongside fusion and perception checks.

## Replay integration

From the repository root, `PYTHONPATH=.:safety python -m avsafety.export` reruns two existing matrix cases: the first successful nominal-GPS seed and the first gated-EKF bias-fault collision seed, both with GPS sigma 1.5 m and normal odometry. It verifies outcome, duration, minimum clearance and localization RMSE against the recorded rows before writing [`viz/public/data/replays`](../viz/public/data/replays).

Each trace carries timestamps, truth and estimated poses, position covariance, known obstacles and goal, sensor-fault regimes and individual injected-event flags, true clearance and terminal outcome. The grey raw-position baseline holds GPS between fixes; it is not dead reckoning. [`replay_integration.json`](../results/replay_integration.json) records the source matrix hash, exact case rows and exported-file hashes. The optional trace does not add random draws or change control actions.
