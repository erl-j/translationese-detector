# Learned vs hand-set rule weights (gran=100)

LR over the engine's per-rule contributions on dev-train (9560 units; C=3 chosen by grouped CV). `multiplier` is the learned coefficient on the hand-set contribution: 1.0 means the hand-set weight is about right, >1 under-weighted, between 0 and 1 over-weighted, **< 0 means the sign flips** (the rule points the wrong way once the other rules are accounted for). `alone AUC` is the rule's contribution used alone as a score on dev-train; `fires` is the share of units with a non-zero contribution. Engine: `tsw.engine:Default`, tsw hash `ae5041cbd85649bf`, commit `589bda5e`.

| rule | family | dir | hand weight | cap | fires | alone AUC | multiplier | std. coef | flag |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `starter.dem_har_vs_denna` | starter | T | 1.00 | 3.00 | 0.776 | 0.477 | +0.611 | +0.246 | alone AUC<0.5 |
| `starter.ska_vs_skall` | starter | T | 0.50 | 1.00 | 0.357 | 0.624 | +48.807 | +2.295 | x48.8 |
| `starter.modal_particles` | starter | O | 1.00 | 3.00 | 1.000 | 0.598 | +0.922 | +0.547 |  |
| `starter.sent_len` | starter | T | 0.50 | 2.00 | 0.992 | 0.654 | +1.441 | +0.538 |  |
| `starter.calques` | starter | T | 1.00 | 3.00 | 0.029 | 0.519 | +0.997 | +0.243 |  |

Learned intercept (prior, log-odds): -0.018 (engine prior +0.000).

