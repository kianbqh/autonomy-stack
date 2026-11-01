# Viewer validation

The local native hardware results are in `results/native-gpu/bench.json`. Three synchronized three-second groups at 10k/100k/1M points completed on RTX 3080 / ANGLE D3D11, Chromium 156.0.8078.4, driver 621.01. Each group retains every frame interval and its actual UTC window. Source-file hashes, episode hash/world and bundle hashes identify the measured version.

Node 24.22.2 typechecking, 16 unit tests and production build passed locally. A later UI-only fix makes End select any terminal timestamp and updates the HUD during capture; it does not change the benchmark renderer or repeat the three valid performance groups. `results/native-ui-final/` records the final normal/collision interactions and screenshots. The terminal collision RMSE is asserted against the exported summary to two decimal places.

The first two local browser attempts stopped before any performance group: an exact accessible-name mismatch, then a collision timestamp between 0.05-second slider steps. Their raw incomplete reports/logs are preserved in the parent execution evidence. The final hardware performance window ran after the other project CPU stages had paused.

The root workflow also exercises the browser using explicitly labeled SwiftShader software WebGL. CI artifacts prove browser correctness in that environment; the README hardware table comes from the local NVIDIA run.
