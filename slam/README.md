# Offline LiDAR odometry and loop closure

I use this module to follow how scan-registration error changes a recorded trajectory, then check what an independently verified loop adds. The front end is a C++17 point-to-plane ICP implementation; Eigen handles matrix algebra, nanoflann supplies kNN queries, and GTSAM 4.2 optimizes the pose graph.

The [Python CI checks](results/ci-6393986/import.json) pass on synthetic trajectories, including an actual evo CLI comparison. Native compilation and the KITTI runs are next; [execution status](results/README.md) tracks their results.

## Registration and bounded mapping

Scans are little-endian float32 XYZI. The loader rejects empty, malformed or nonfinite input. Range filtering precedes an own voxel-centroid filter. Voxel cells are ordered lexicographically and, if the configured cap is exceeded, uniformly selected by ordered cell index. This is deterministic for the same input bytes.

Local PCA normals use kNN, a radius limit and a smallest-to-middle eigenvalue planarity gate. ICP matches source points to the bounded map, rejects distant pairs, retains the configured residual fraction, and solves Huber-weighted point-to-plane normal equations. Six-DoF eigenvalue/condition gates reject locally ambiguous geometry, including an isolated plane. Failed convergence, a bad solve, too few inliers or failed fitness stops the attempt and retains its journal; no fallback pose is silently counted as a successful frame. `rmse_m` is the unweighted retained-correspondence point-to-plane fitness, after convergence.

The initial guess is constant velocity: `X_(i-1) * (inverse(X_(i-2)) * X_(i-1))`. Keyframes trigger at 1.5 m or 10 degrees of total relative rotation. The target contains at most 15 recent keyframes and 100,000 downsampled map points. A scan/loop cloud has at most 35,000 points. Historical descriptors, indices and graph poses have a 4,000-keyframe guard; reaching it fails explicitly. Old loop clouds are loaded from their original scan files when needed instead of retaining every scan in memory.

The own polar ring/sector max-height descriptor follows the Scan Context idea. Ring-key distance nominates five temporally eligible candidates; the descriptor checks every sector yaw shift. Low angular support and a nearly tied nonadjacent yaw fail the frozen descriptor gates. Candidates then undergo actual forward/reverse ICP, fitness/inlier/conditioning/convergence gates and a reciprocal transform check. Every nominated candidate and temporal exclusion is retained in `loops.jsonl` and `loops.json`. No GT path is accepted by the C++ application.

The graph has an anchored first pose, Gaussian odometry factors and Huber loop factors using `BetweenFactor<Pose3>`. It runs **one end-of-sequence Levenberg–Marquardt optimization**, not iSAM2. Noise values are declared modeling choices, not calibrated covariance estimates. The front end always uses its odometry map; the graph never improves registration retrospectively.

## Coordinate convention

Column vectors; `T_A_B` maps B coordinates into A. Internally `X_i=T_L0_Li` and `X_0=I`. The point-to-plane residual is `n·(T*p-q)`. Twists are `[omega, v]`. For numerical conditioning, ICP subtracts the deterministic target centroid c, solves `T_center_source=Translate(-c)*T_target_source`, and applies a **left** update `T_center_source <- Exp(delta)*T_center_source` with Jacobian `[transformed_p cross n, n]` in that centered frame. It returns `Translate(c)*T_center_source`. In the original target coordinates this update is conjugated by Translate(c); a distant arbitrary world origin must not masquerade as degenerate geometry. Translation in Exp uses the SO(3) left Jacobian. Graph measurement `Z_i_j=inverse(X_i)*X_j` maps scan j into scan i.

KITTI odometry `Tr=T_rectCam0_Lidar` already includes rectification. Camera0-relative output is `Tr * X_i * inverse(Tr)`. Official GT is normalized by `inverse(GT_0)*GT_i`. Image_2 projection is **`P2 * Tr * p_L`**; no additional raw-KITTI `R0_rect` is multiplied.

All scan poses are retained. Between keyframes a,b, interpolate graph corrections on SE(3): `C_k=X_k_optimized*inverse(X_k_odometry)`, `C_i=C_a*Exp(alpha*Log(inverse(C_a)*C_b))`, output `C_i*X_i`. Alpha uses the actual `times.txt` values. Past the final keyframe, hold its correction. This is a documented correction-interpolation approximation, not a new sensor measurement. Output rows and timestamps match the original scans exactly.

## Setup after resource release

Use an isolated WSL build root on the G-backed Linux disk, for example `/opt/codex-resume/all-modules-v2/m3/slam`. Keep the shared Python environment read-only. Prerequisites are a C++17 compiler, CMake >=3.22, Ninja, and compatible Boost headers/libraries. Do not install them or start a build during another module's reserved timing window.

[`dependencies.lock.json`](dependencies.lock.json) now records the actual official archive bytes/SHA-256 and inspected version/license files for Eigen 3.4.0, nanoflann 1.6.3, GTSAM 4.2 and GoogleTest 1.15.2. The original candidate remains preparation history. GTSAM's exact upstream `4.2` tag declares numeric version 4.2.0 and retains a display suffix `a9`; the source archive is unchanged. The build helper refuses unresolved candidate locks. CMake never fetches dependencies automatically.

The local prerequisite setup uses GCC 13.3.0, CMake 4.4.4, Ninja 1.13.2 and a private Boost 1.83.0 prefix. All nine Boost components were actually discovered, linked and loaded: serialization, system, filesystem, thread, program_options, date_time, timer, chrono and regex. [`prerequisites.lock.json`](prerequisites.lock.json) pins the official Boost archive and isolated build recipe. A default GTSAM configure failed on CMake 4's removed pre-3.5 compatibility; `--cmake-policy-version-minimum 3.5` explicitly supplies that compatibility policy to GTSAM without editing its source. This prerequisite evidence is separate from the still-pending GTSAM/SLAM build and native tests.

```bash
# From the repository root, after committing the clean source/locks.
# Put CMake/Ninja in this child PATH; do not install into a shared environment.
SLAM_ROOT=/opt/codex-resume/all-modules-v2/m3/slam
python3 slam/python/fetch_locked_sources.py --lock slam/dependencies.lock.json \
  --output "$SLAM_ROOT/archives"
python3 slam/python/build_native.py --lock slam/dependencies.lock.json \
  --archives-dir "$SLAM_ROOT/archives" --scratch-parent "$SLAM_ROOT/builds" \
  --repo-root "$PWD" --jobs 1 \
  --prerequisite-prefix "$SLAM_ROOT/boost-1.83.0-prefix" \
  --cmake-policy-version-minimum 3.5
```

Use `--git /explicit/native/git` if necessary. `--prerequisite-prefix` is repeatable; the new build's own prefix comes first, so GTSAM and SLAM use the same private Eigen. The helper installs only under its unique private prefix, compiles the library/CLI, executes real GoogleTests and writes source/compiler/dependency/binary/tool identities and resolved dynamic libraries. Its formal execution is still pending.

The isolated CPython 3.12/Linux x86_64 evaluator has installed/imported NumPy 2.2.6, SciPy 1.15.3, OpenCV-headless 4.11.0.86, matplotlib 3.10.3, evo 1.31.0 and pytest 8.3.5; `pip check` passes. [`evaluation-environment.lock.json`](evaluation-environment.lock.json) and [`requirements-eval.lock.txt`](requirements-eval.lock.txt) freeze all 30 actual direct/transitive runtime packages and their artifact hashes. To reproduce in a new isolated environment on that target, install with `python -m pip install --require-hashes -r slam/requirements-eval.lock.txt`. The shared tool environment remains read-only.

The unified [tests workflow](../.github/workflows/tests.yml) adds `slam-native-cpu`: the four locked sources, 13 native tests, positive/negative synthetic fixtures and hash-locked evaluator checks, retaining small receipt artifacts. Its remote Ubuntu apt Boost installation is a separate environment from the local private Boost prefix above. Existing job bodies are preserved; `ci/slam-job.candidate.yml` remains preparation history. The new job's actual CI results are still pending.

## Official data and selective import

KITTI odometry is recorded driving data by Geiger et al., licensed [CC BY-NC-SA 3.0](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). Acquire the official odometry Velodyne, poses, calibration and color ZIPs through the provider's authorized distribution. Preserve the original archives and their completed download receipts outside Git. No raw scans, full images, archives or large clouds belong in this repository.

Sequence **07 is mandatory: 1,101 contiguous scans/timestamps/poses**. Sequence **00 is stretch: 4,541** and uses the same algorithm configuration. The fixed projection frames for 07 are 0, 550 and 1100. `import_kitti.py` validates completed archive byte lengths and SHA-256 receipts, member traversal/symlinks/duplicates, selected member CRC and extracted hashes. It imports only the chosen sequence plus those 2–3 image_2 frames; unchanged verified extractions are reused. It has no network action and never uses `extractall`.

```bash
python slam/python/import_kitti.py --archives /path/on/G/completed-archives.json \
  --config slam/configs/seq07.json --output /path/on/G/kitti-seq07-verified
```

The archive receipt schema is `{"archives":[{"path":"...zip","bytes":123,"sha256":"actual..."}, ...]}` with exactly the four official names. `123` describes the schema only; use actual completed receipts, never an example number. The importer fails on incomplete `.part` files. Sequence 00 additionally requires `--allow-sequence-00` and its own declared frame/image selections.

## One-command offline run

Inside WSL/Linux, from the repository root, provide the verified local archive manifest, the actual frozen build receipt/binary, isolated evaluator interpreter and a G-backed output parent:

```bash
bash slam/scripts/run_kitti.sh 07 \
  --archives-manifest "$SLAM_ROOT/completed-archives.json" \
  --build-receipt /private/build/build.json \
  --binary /private/build/slam-build/run_kitti_slam \
  --eval-python "$SLAM_ROOT/eval-venv/bin/python" \
  --output-parent "$SLAM_ROOT/pipelines" \
  --git /explicit/native/git \
  --official-oracle /private/accepted/official_oracle
```

The thin entrypoint calls the existing import → native run → evo evaluation (plus the supplied official oracle) → calibration diagnostic → replay export CLIs. Each pipeline attempt is new; downstream steps use only the exact paths returned by that invocation. It never searches for a latest run, overwrites an old result, downloads/installs dependencies, or changes algorithm settings from GT. `--data-root /private/verified-extraction` optionally revalidates/reuses an existing extraction. Failures exit nonzero and retain `pipeline.json`, per-stage logs and the original runner's attempt evidence. No formal run or browser acceptance has been performed through this entrypoint yet.

Every required path also has an environment fallback: `M3_SLAM_ARCHIVES_MANIFEST`, `M3_SLAM_BUILD_RECEIPT`, `M3_SLAM_BINARY`, `M3_SLAM_EVAL_PYTHON`, `M3_SLAM_OUTPUT_PARENT`; optional `M3_SLAM_GIT` and `M3_SLAM_OFFICIAL_ORACLE` cover the two tool paths. `bash slam/scripts/run_kitti.sh --help` lists the options. Sequence defaults to 07. Stretch00 needs both `--allow-sequence-00` and an explicit full00 config; no00 config or run is supplied here. The accepted official adapter is full07 only. Omitting the optional oracle keeps the pipeline receipt explicitly **partial**, even when all other stages succeed; it cannot establish official parity or complete acceptance.

## Runs and evidence

Commit clean `slam/` source before building/running. All runners require the real Git executable; use `--git /explicit/native/git` if the WSL distribution lacks Git. They do not fabricate source SHAs or mutate global identity. Every attempt gets a unique output directory; failures retain configuration, command, UTC/monotonic timing, process handle, stdout/stderr and failure information. Run records include source/config/input hashes, package versions, CPU/platform, native build receipt and binary hash. Raw point data stay outside Git; small derived trajectories, metric JSON/CSV and attributed plots can be retained.

```bash
python slam/python/run_experiment.py --config slam/configs/seq07.json \
  --input-manifest /path/on/G/kitti-seq07-verified/input-manifest.json \
  --binary /private/build/slam-build/run_kitti_slam \
  --build-receipt /private/build/build.json --repo-root "$PWD" \
  --output-parent /path/on/G/slam-runs
python slam/python/eval_traj.py --config slam/configs/seq07.json \
  --native-run /path/on/G/slam-runs/UNIQUE/native \
  --truth /path/on/G/kitti-seq07-verified/dataset/poses/07.txt \
  --repo-root "$PWD" --output-parent /path/on/G/slam-evaluation --evo
python slam/python/calib_check.py --config slam/configs/seq07.json \
  --input-manifest /path/on/G/kitti-seq07-verified/input-manifest.json \
  --repo-root "$PWD" --output-parent /path/on/G/slam-calibration
```

`run_experiment.py` gates exact formal frame counts and verifies extracted hashes before timing. It reports raw per-frame/stage wall samples, process CPU time and mean/p95 using linear percentile interpolation. Registration-only time, frame I/O-inclusive time, final graph optimization and whole-process wall time are separate. Frame time excludes final diagnostic flush; whole-process wall includes it. There are no warmups or repeated sequence passes. Background work must be recorded externally; no exclusive-machine timing claim is justified while other workloads compete.

Evaluation reports no-scale SE(3) Umeyama-aligned position ATE RMSE, 1-frame RPE, 100 m RPE and KITTI-style 100–800 m segment errors. Distance endpoints are the first GT cumulative distance **strictly greater** than start+length; RPE uses all start frames, segment evaluation uses every tenth frame. The segment calculation follows the [official devkit](https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_odometry.zip): float32 distances and scalar errors, double pose matrices, start-frame/length ordering and a serial mean over all valid segments. ATE and RPE retain double precision. Translation is percent; rotation retains rad/m, deg/m and explicitly converted deg/100 m. Empty segment sets have count 0 and null means. Native devkit comparison on the recorded trajectories is still pending.

The external evo 1.31.0 CLI independently checks ATE and translation/rotation RPE. The 100 m comparison expands the exact selected original pairs into a CLI input, retaining even-index adjacent errors and recording ignored pair-boundary errors. It saves all commands, ZIP arrays, stdout/stderr, pair counts and per-sample differences. No GPL evo source is copied into this C++/Python implementation. An absent/failed independent check leaves evaluation status partial. Loop true/false-positive labels use GT separation <4 m **only after the run**, never for loop selection or tuning. Trajectory/error figures show accepted edges with those post-hoc labels.

## Recorded-data replay and failed attempts

The replay format, `kitti-slam-replay/v1`, preserves full camera0-relative SE(3), original timestamps, separate ICP/optimized/GT tracks and every loop decision. A failed attempt retains its accepted odometry prefix. Prefix checkpoint flush is included in frame-inclusive timing, while registration-only timing is unchanged.

```bash
python slam/python/export_replay.py --run /path/on/G/slam-runs/UNIQUE \
  --truth /path/on/G/kitti-seq07-verified/dataset/poses/07.txt \
  --repo-root "$PWD" --output-parent /path/on/G/slam-replay-exports
```

Open the separate [3D viewer](../viz/README.md#recorded-slam-entry) with `?view=slam&replay=<same-origin-json-url>`. TypeScript and the viewer build pass in CI. Playback of a real KITTI export and its browser checks are still pending.

The existing 2D simulator's sensor-noise and sampling checks pass in the same Python CI run. They exercise the simulator's synchronous Gaussian noise, bias random walk and GPS outliers; KITTI sensor calibration is evaluated separately.

## Camera–LiDAR diagnostic

Projection applies positive-depth/range/image masks and a nearest-depth z buffer. The metric compares LiDAR depth-discontinuity pixels against a fixed Canny-edge distance transform. It caps/normalizes pixel distances, reports projected and edge support, applies fixed minimum support gates and penalizes low in-image point coverage. Sparse holes alone are not depth discontinuities. Left perturbations in rectified camera0 axes sweep each rotation over ±2 degrees and each translation over ±0.1 m. Settings and frame selection are frozen in `configs/seq07.json` before scores exist.

The report keeps all frame/offset scores, curve shapes, frame-bootstrap intervals, boundary optima and whether the supplied calibration actually wins. It makes 2–3 depth-colored overlays with KITTI attribution. A losing supplied transform is a valid result. This checks sensor consistency only; it does not establish calibrated extrinsics or targetless-estimation recovery.

## Checks and limitations

```bash
python -B -m unittest discover -s slam/tests -p test_static_contracts.py -v
python -m pytest --capture=sys -q slam/tests/test_eval.py
PYTHONPATH=. python -m pytest --capture=sys -q slam/tests/test_sim_sensor_models.py
ctest --test-dir /private/build/slam-build --output-on-failure
```

The C++ tests cover a nondegenerate multi-plane known SE(3), noise/partial overlap/outliers, single-plane degeneracy, left-update finite differences, descriptor yaw sign, actual ICP-verified toy-loop improvement, timestamp correction, three-frame binary I/O and the rectified calibration chain. Numerical Python tests check no-scale alignment, deliberately wrong scale, distance endpoints, null short-sequence metrics and projection occlusion. These are engineering fixtures, not recorded driving results.

This is offline LiDAR-only SLAM without IMU fusion, deskewing, dynamic-object segmentation or a real vehicle deployment. PCA normals, descriptor retrieval, initial odometry drift and local ICP minima can fail. Local conditioning/reciprocal tests do not prove globally unique alignment; repetitive scenes can still yield false loops. Correction interpolation is an approximation. Fixed factor sigmas are not learned uncertainties. CPU performance, dataset completeness, native API compatibility, official/evo metric parity and calibration behavior remain unverified until the listed runs finish.

References: [GTSAM 4.2](https://github.com/borglab/gtsam/releases/tag/4.2), [nanoflann 1.6.3](https://github.com/jlblancoc/nanoflann/releases/tag/v1.6.3), [Scan Context authors](https://github.com/gisbi-kim/scancontext_tro), [KITTI odometry conventions and evaluation](https://www.cvlibs.net/datasets/kitti/eval_odometry.php), [KITTI-linked pykitti loader](https://github.com/utiasSTARS/pykitti/blob/master/pykitti/odometry.py), [evo 1.31.0](https://github.com/MichaelGrupp/evo/releases/tag/v1.31.0). Descriptor/ICP code here is independently written; the named libraries retain their own licenses.
