# Measured SLAM result

| Metric | Odometry | SLAM | Unit |
|---|---:|---:|---|
| ATE (SE3, no scale) | 0.56306443 | 0.370272684 | m |
| RPE one frame, translation | 0.0174067442 | 0.0174025719 | m |
| RPE one frame, rotation | 0.0609375168 | 0.0609270764 | deg |
| RPE 100m, translation | 0.765795786 | 0.790350799 | m |
| RPE 100m, rotation | 0.756455196 | 0.748728228 | deg |
| Official KITTI translation | 0.688189128 | 0.681179855 | % |
| Official KITTI rotation | 0.510249857 | 0.508066856 | deg / 100m |

1101 original KITTI07 frames, 381 keyframes, 2 accepted loops;
post-hoc loop labels: 2 true positives and 0 false positives.
Recall over all possible revisits is not defined by this candidate-only log.
ATE improves; 100m translation RPE increases and is retained without tuning.
The 100m RPE uses 881 explicit first-distance-greater pairs. The official KITTI
aggregate uses 317 valid 100..800m segments and the original float32 scalar rule.
Both tracks passed all five evo CLI crosschecks and the fixed official oracle.

Native generation: `61d485e79d8f1cd4ef5791f148c853555f4db390`; corrected evaluator:
`75bb06e7b114a2f004d0d8c44348864e728095f8`. Actual runtime configuration:
`ee689925bcecd2ee7c71817e47903a2c677c7c8dd09684ef2c7e5067d9d6c5da`. The default configuration contains these exact tested bytes. Original strict configuration 51a4 and failures at 27/50/114 remain.
The corrected result is separate from evaluation-original-partial; nothing in
that original failed parity attempt was rewritten.

Native process timing is 36.756231s on a shared
Windows/WSL host. Raw frame timings are included; this is not an isolated or
repeated benchmark. Calibration is independently measured with original 51a4,
whose calibration fields are identical to the runtime configuration. Three
nominal overlays are valid; mean consistency score is .5733597079912821. The
one-axis tz sweep best at +0.025 m is not calibration recovery and was not applied.

Read ATTRIBUTION_AND_SCOPE.md for KITTI attribution and dataset restrictions.
