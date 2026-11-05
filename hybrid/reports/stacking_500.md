# Learned vs hand-set rule weights (gran=500)

LR over the engine's per-rule contributions on trn (2734 units; C=0.03 chosen by session-grouped CV inside trn). `multiplier` is the learned coefficient on the hand-set contribution: 1.0 means the hand-set weight is about right, >1 under-weighted, between 0 and 1 over-weighted, **< 0 means the sign flips** (the rule points the wrong way once the other rules are accounted for). `alone AUC` is the rule's contribution used alone as a score on trn; `fires` is the share of units with a non-zero contribution. Engine: `tsw.engine:Default`, tsw hash `263693870a139a72`, commit `e1dd1e9b`.

| rule | family | dir | hand weight | cap | fires | alone AUC | multiplier | std. coef | flag |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `starter.dem_har_vs_denna` | starter | T | 1.00 | 3.00 | 0.996 | 0.505 | +0.471 | +0.403 | x0.5 |
| `starter.ska_vs_skall` | starter | T | 0.50 | 1.00 | 0.802 | 0.754 | +2.400 | +0.229 | x2.4 |
| `starter.modal_particles` | starter | O | 1.00 | 3.00 | 1.000 | 0.672 | +0.725 | +0.820 |  |
| `starter.sent_len` | starter | T | 0.50 | 2.00 | 0.996 | 0.781 | +1.080 | +1.073 |  |
| `starter.calques` | starter | T | 1.00 | 3.00 | 0.116 | 0.566 | +0.715 | +0.363 |  |

Learned intercept (prior, log-odds): -0.208 (engine prior +0.000).

