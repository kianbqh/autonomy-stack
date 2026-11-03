# Paired nominal success

Each seed uses the same 100 held-out resets. Standard deviation is the population standard deviation across the three trained-seed success rates.

| Method | Successes by seed /100 | Mean | Population std (percentage points) |
|---|---|---:|---:|
| BC | 5 / 4 / 7 | 5.33% | 1.25 |
| DR-BC | 4 / 9 / 22 | 11.67% | 7.59 |
| Learned MPC | 6 / 5 / 7 | 6.00% | 0.82 |

Learned MPC underperformed DR-BC on mean success. Horizons, seeds, controller settings and test splits were fixed before evaluation.

[Original source-generated comparison](comparison.md) · [Paired terminal outcomes](paired-policy-outcomes.csv)
