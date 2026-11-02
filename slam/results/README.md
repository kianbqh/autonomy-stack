# Execution status

No KITTI or native C++ result has been produced. This directory contains no measured accuracy, loop-closure or timing claim.

| Planned run | Status | Required acceptance |
|---|---|---|
| C++ CPU unit fixtures | not run | Pinned dependencies, real compile, GoogleTest pass |
| KITTI 07 odometry + loop graph | not run | All 1,101 official scans and timestamps, immutable run evidence |
| KITTI 00 stretch | not run | All 4,541 scans; identical algorithm settings |
| evo independent metrics | not run | Real evo 1.31.0 CLI, raw ZIP/logs, exact pair reconciliation |
| Official KITTI devkit parity | not run | Pin provider source, compare segment IDs and aggregation |
| Camera–LiDAR overlays/sweeps | not run | Predeclared frames 0, 550, 1100; actual images/scans and support gates |
| Recorded SLAM viewer | not run | Real export, separate 3D parser/entry, typecheck/build/browser and failure-prefix acceptance |
| Explicit shared-2D noise/schedule checks | not run | Four real RobotSim numerical tests; no transport-latency claim |

Synthetic unit fixtures test engineering behavior only. They cannot fill any KITTI row.
