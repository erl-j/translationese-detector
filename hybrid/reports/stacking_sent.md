# Learned vs hand-set rule weights (gran=sent)

LR over the engine's per-rule contributions on dev-train (58721 units; C=3 chosen by grouped CV). `multiplier` is the learned coefficient on the hand-set contribution: 1.0 means the hand-set weight is about right, >1 under-weighted, between 0 and 1 over-weighted, **< 0 means the sign flips** (the rule points the wrong way once the other rules are accounted for). `alone AUC` is the rule's contribution used alone as a score on dev-train; `fires` is the share of units with a non-zero contribution. Engine: `tsw.engine:Default`, tsw hash `ae5041cbd85649bf`, commit `8ccbf52b`.

| rule | family | dir | hand weight | cap | fires | alone AUC | multiplier | std. coef | flag |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `starter.dem_har_vs_denna` | starter | T | 1.00 | 3.00 | 0.258 | 0.476 | +0.804 | +0.131 | alone AUC<0.5 |
| `starter.ska_vs_skall` | starter | T | 0.50 | 1.00 | 0.083 | 0.528 | +77.518 | +1.766 | x77.5 |
| `starter.modal_particles` | starter | O | 1.00 | 3.00 | 1.000 | 0.584 | +0.940 | +0.229 |  |
| `starter.sent_len` | starter | T | 0.50 | 2.00 | 0.995 | 0.570 | +2.148 | +0.294 | x2.1 |
| `starter.calques` | starter | T | 1.00 | 3.00 | 0.005 | 0.504 | +1.034 | +0.105 |  |

Learned intercept (prior, log-odds): -0.079 (engine prior +0.000).

