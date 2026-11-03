# Learned-dynamics MPC comparison

All outcomes are actual native FetchPush step50 success on paired held-out resets.

| Method | Success mean | Population std | Episodes per seed |
|---|---:|---:|---:|
| bc | 0.05333333333333334 | 0.012472191289246475 | 100 |
| bc_dr | 0.11666666666666665 | 0.07586537784494028 | 100 |
| learned_mpc | 0.06 | 0.008164965809277263 | 100 |

Negative results

Learned MPC mean success (0.06) was lower than bc_dr (0.11666666666666665).
