# Learned vs hand-set rule weights (gran=500)

LR over the engine's per-rule contributions on dev-train (1823 units; C=0.03 chosen by grouped CV). `multiplier` is the learned coefficient on the hand-set contribution: 1.0 means the hand-set weight is about right, >1 under-weighted, between 0 and 1 over-weighted, **< 0 means the sign flips** (the rule points the wrong way once the other rules are accounted for). `alone AUC` is the rule's contribution used alone as a score on dev-train; `fires` is the share of units with a non-zero contribution. Engine: `tsw.engine:Default`, tsw hash `ae5041cbd85649bf`, commit `589bda5e`.

| rule | family | dir | hand weight | cap | fires | alone AUC | multiplier | std. coef | flag |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `starter.dem_har_vs_denna` | starter | T | 1.00 | 3.00 | 0.994 | 0.486 | +0.348 | +0.283 | x0.3; alone AUC<0.5 |
| `starter.ska_vs_skall` | starter | T | 0.50 | 1.00 | 0.813 | 0.757 | +3.652 | +0.385 | x3.7 |
| `starter.modal_particles` | starter | O | 1.00 | 3.00 | 1.000 | 0.699 | +0.713 | +0.876 |  |
| `starter.sent_len` | starter | T | 0.50 | 2.00 | 0.996 | 0.759 | +0.897 | +0.893 |  |
| `starter.calques` | starter | T | 1.00 | 3.00 | 0.118 | 0.575 | +0.775 | +0.398 |  |

Learned intercept (prior, log-odds): -0.292 (engine prior +0.000).

