# Data, results and scope

The complete KITTI odometry 07 run contains 1,101 frames. Native generation
used commit `61d485e79d8f1cd4ef5791f148c853555f4db390`; evaluator commit
`75bb06e7b114a2f004d0d8c44348864e728095f8` passed both evo and the fixed
official KITTI cross-check. The [final evaluation](evaluation-corrected-75bb/metrics.json)
records both identities. See the [measured results](REAL_METRICS_SUMMARY.json)
for all metrics, including the higher 100 m translation RPE after loop closure.

The [earlier partial evaluation](evaluation-original-partial/metrics.json) is
retained alongside the corrected result. It exposed a metric-definition
difference on rounded rotation matrices; the correction preserved the original
comparison tolerances and official KITTI calculations. The native trajectories
were reused without another SLAM run.

The [tested configuration](configs/seq07-actual-ee689925.json) uses uniform
1 mm translation and 0.0001 rad rotation stopping precision. The
[original configuration](configs/seq07-original-51a4.json) and failed attempts
are retained. Fitness, inlier, conditioning and loop gates stayed unchanged.
Timing comes from one shared Windows/WSL host run, without warmup or repeats;
it is not an isolated benchmark.

The [calibration analysis](calibration-original51a4/calibration.json) uses
recorded frames 0, 550 and 1100. Its calibration settings are identical in both
configurations. These fixed one-axis sweeps measure sensor consistency, not
calibration recovery. The best tz offset of +0.025 m was not applied; SLAM used
the supplied extrinsic calibration unchanged.

Dataset: [KITTI odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php),
A. Geiger, P. Lenz, R. Urtasun and collaborators. KITTI data is licensed under
[CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/).
The overlays derive from the official color images and LiDAR scans at the
three named frames. Raw dataset archives and scans are not included. The
external official-oracle executable and third-party implementation sources
are also excluded.
