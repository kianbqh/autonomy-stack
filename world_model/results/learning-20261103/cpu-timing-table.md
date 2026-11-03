# Shared-host CPU control timing

The table summarizes 15,000 measured controls, excluding five warmup controls per model seed. These are CPU wall times on a host shared with ordinary user workloads.

| Timer scope | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|
| planner_ms | 3.342 | 3.424 | 4.375 | 5.096 |
| host_policy_ms | 3.409 | 3.490 | 4.461 | 5.200 |
| policy_act_ms | 3.483 | 3.564 | 4.547 | 5.295 |
| full_control_step_ms | 4.556 | 4.631 | 5.731 | 6.553 |

[Raw controls](raw-timings.csv) and [exact timer definitions](summary.json) retain the observed samples and exclusions.
