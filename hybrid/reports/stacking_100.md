# Learned vs hand-set rule weights (gran=100)

LR over the engine's per-rule contributions on trn (14170 units; C=0.003 chosen by session-grouped CV inside trn). `multiplier` is the learned coefficient on the hand-set contribution: 1.0 means the hand-set weight is about right, >1 under-weighted, between 0 and 1 over-weighted, **< 0 means the sign flips** (the rule points the wrong way once the other rules are accounted for). `alone AUC` is the rule's contribution used alone as a score on trn; `fires` is the share of units with a non-zero contribution. Engine: `tsw.engine:Default`, tsw hash `263693870a139a72`, commit `e1dd1e9b`.

| rule | family | dir | hand weight | cap | fires | alone AUC | multiplier | std. coef | flag |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `starter.dem_har_vs_denna` | starter | T | 1.00 | 3.00 | 0.775 | 0.484 | +0.595 | +0.253 | alone AUC<0.5 |
| `starter.ska_vs_skall` | starter | T | 0.50 | 1.00 | 0.348 | 0.623 | +2.002 | +0.091 | x2.0 |
| `starter.modal_particles` | starter | O | 1.00 | 3.00 | 1.000 | 0.589 | +0.730 | +0.393 |  |
| `starter.sent_len` | starter | T | 0.50 | 2.00 | 0.992 | 0.675 | +1.516 | +0.565 |  |
| `starter.calques` | starter | T | 1.00 | 3.00 | 0.029 | 0.517 | +0.799 | +0.189 |  |

Learned intercept (prior, log-odds): -0.056 (engine prior +0.000).

