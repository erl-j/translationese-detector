# Learned vs hand-set rule weights (gran=sent)

LR over the engine's per-rule contributions on trn (87904 units; C=0.003 chosen by session-grouped CV inside trn). `multiplier` is the learned coefficient on the hand-set contribution: 1.0 means the hand-set weight is about right, >1 under-weighted, between 0 and 1 over-weighted, **< 0 means the sign flips** (the rule points the wrong way once the other rules are accounted for). `alone AUC` is the rule's contribution used alone as a score on trn; `fires` is the share of units with a non-zero contribution. Engine: `tsw.engine:Default`, tsw hash `263693870a139a72`, commit `e1dd1e9b`.

| rule | family | dir | hand weight | cap | fires | alone AUC | multiplier | std. coef | flag |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `starter.dem_har_vs_denna` | starter | T | 1.00 | 3.00 | 0.257 | 0.475 | +0.841 | +0.142 | alone AUC<0.5 |
| `starter.ska_vs_skall` | starter | T | 0.50 | 1.00 | 0.080 | 0.527 | +2.599 | +0.055 | x2.6 |
| `starter.modal_particles` | starter | O | 1.00 | 3.00 | 1.000 | 0.590 | +0.873 | +0.191 |  |
| `starter.sent_len` | starter | T | 0.50 | 2.00 | 0.996 | 0.578 | +2.195 | +0.297 | x2.2 |
| `starter.calques` | starter | T | 1.00 | 3.00 | 0.005 | 0.504 | +0.924 | +0.093 |  |

Learned intercept (prior, log-odds): -0.015 (engine prior +0.000).

