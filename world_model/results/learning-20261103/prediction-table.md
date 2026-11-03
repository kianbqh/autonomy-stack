# Recorded-action prediction errors

Values average episode means, then average over the three independent model seeds. Units are millimeters.

| Horizon | Learned object | CV object | Learned gripper | CV gripper |
|---:|---:|---:|---:|---:|
| 1 | 23.401 | 3.446 | 26.971 | 11.635 |
| 5 | 38.004 | 38.854 | 37.786 | 81.635 |
| 10 | 57.767 | 100.667 | 56.708 | 181.201 |
| 20 | 93.031 | 211.723 | 97.525 | 377.150 |
| 30 | 126.085 | 284.348 | 126.898 | 562.960 |

The learned model is worse than constant velocity at one-step object prediction in this run; lower long-horizon error did not establish better control than DR-BC.
