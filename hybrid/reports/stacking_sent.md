# Learned vs hand-set rule weights (gran=sent)

LR over the engine's per-rule contributions on trn (87904 units; C=0.01 chosen by session-grouped CV inside trn). `multiplier` is the learned coefficient on the hand-set contribution: 1.0 means the hand-set weight is about right, >1 under-weighted, between 0 and 1 over-weighted, **< 0 means the sign flips** (the rule points the wrong way once the other rules are accounted for). `alone AUC` is the rule's contribution used alone as a score on trn; `fires` is the share of units with a non-zero contribution. Engine: `tsw.engine:Default`, tsw hash `00a590c8e3449552`, commit `cef0b257`.

| rule | family | dir | hand weight | cap | fires | alone AUC | multiplier | std. coef | flag |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `calques.lexicon` | calques | T | 1.00 | 3.00 | 0.036 | 0.525 | +1.777 | +0.362 |  |
| `calques.native_counterparts` | calques | O | 1.00 | 2.00 | 0.001 | 0.501 | +1.641 | +0.080 |  |
| `calques.forsoka_att` | calques | T | 1.00 | 1.50 | 0.010 | 0.500 | +0.959 | +0.049 |  |
| `calques.pied_piped_relative` | calques | T | 1.00 | 1.50 | 0.011 | 0.505 | +0.852 | +0.072 |  |
| `connpunct.native_connectives` | connpunct | O | 0.50 | 2.00 | 1.000 | 0.597 | +1.967 | +0.224 |  |
| `connpunct.additive_ocksa` | connpunct | O | 0.40 | 1.50 | 1.000 | 0.600 | +2.427 | +0.237 | x2.4 |
| `connpunct.calque_connectives` | connpunct | T | 0.80 | 2.00 | 1.000 | 0.438 | +0.968 | +0.160 | alone AUC<0.5 |
| `connpunct.att_drop` | connpunct | O | 1.00 | 1.50 | 1.000 | 0.582 | +0.676 | +0.026 |  |
| `connpunct.som_drop` | connpunct | T | 0.70 | 1.50 | 0.018 | 0.503 | +1.081 | +0.047 |  |
| `connpunct.initial_och` | connpunct | T | 1.00 | 2.00 | 1.000 | 0.420 | +1.245 | +0.093 | alone AUC<0.5 |
| `connpunct.initial_men` | connpunct | T | 0.70 | 1.50 | 1.000 | 0.427 | +2.490 | +0.124 | x2.5; alone AUC<0.5 |
| `connpunct.initial_adverbial_comma` | connpunct | T | 1.00 | 2.00 | 0.094 | 0.511 | +0.771 | +0.105 |  |
| `connpunct.clause_comma_omission` | connpunct | T | 0.70 | 1.50 | 0.053 | 0.504 | +1.140 | +0.090 |  |
| `connpunct.written_out_abbrev` | connpunct | T | 0.50 | 1.50 | 0.025 | 0.511 | +1.871 | +0.139 |  |
| `connpunct.parenthetical_dash` | connpunct | T | 0.50 | 1.50 | 1.000 | 0.443 | +1.257 | +0.119 | alone AUC<0.5 |
| `connpunct.explanatory_parens` | connpunct | T | 1.00 | 1.00 | 1.000 | 0.422 | +0.799 | +0.047 | alone AUC<0.5 |
| `connpunct.exclamation` | connpunct | O | 0.50 | 1.50 | 1.000 | 0.582 | +1.726 | +0.083 |  |
| `starter.modal_particles` | starter | O | 1.00 | 3.00 | 1.000 | 0.591 | +0.965 | +0.244 |  |
| `starter.sent_len` | starter | T | 0.50 | 2.00 | 0.996 | 0.578 | +1.844 | +0.250 |  |
| `particles.namligen` | particles | O | 1.00 | 2.50 | 1.000 | 0.585 | +1.309 | +0.207 |  |
| `particles.anda` | particles | O | 1.00 | 2.00 | 1.000 | 0.584 | +0.975 | +0.077 |  |
| `particles.alltsa` | particles | O | 1.00 | 1.50 | 1.000 | 0.584 | +1.112 | +0.093 |  |
| `particles.stance` | particles | O | 1.00 | 2.00 | 1.000 | 0.584 | +0.920 | +0.104 |  |
| `particles.da_anaphoric` | particles | O | 1.00 | 2.00 | 1.000 | 0.582 | +1.000 | +0.047 |  |
| `particles.of_course` | particles | T | 1.00 | 1.50 | 0.015 | 0.500 | +1.188 | +0.051 |  |
| `particles.dvs` | particles | O | 1.00 | 1.50 | 1.000 | 0.586 | +1.087 | +0.120 |  |
| `particles.particle_verbs` | particles | O | 1.00 | 1.50 | 1.000 | 0.583 | +0.732 | +0.065 |  |
| `particles.ta_itu_med` | particles | T | 1.00 | 2.00 | 1.000 | 0.425 | +1.026 | +0.135 | alone AUC<0.5 |
| `particles.generic_man` | particles | O | 0.25 | 1.50 | 1.000 | 0.587 | +2.551 | +0.068 | x2.6 |
| `particles.copular_det` | particles | O | 0.50 | 1.50 | 1.000 | 0.582 | +1.507 | +0.080 |  |
| `particles.poss_1st` | particles | T | 0.60 | 1.50 | 1.000 | 0.462 | +1.439 | +0.122 | alone AUC<0.5 |
| `particles.poss_3rd` | particles | T | 0.80 | 2.00 | 1.000 | 0.445 | +0.938 | +0.160 | alone AUC<0.5 |
| `particles.han_hon` | particles | T | 0.50 | 1.00 | 1.000 | 0.436 | +1.709 | +0.132 | alone AUC<0.5 |
| `particles.ni_vs_man` | particles | T | 1.00 | 1.50 | 0.077 | 0.518 | +0.616 | +0.084 |  |
| `register.kommer_att_vs_ska` | register | T | 1.00 | 2.50 | 0.136 | 0.550 | +0.939 | +0.298 |  |
| `register.skulle_vilja_vs_vill` | register | T | 1.00 | 2.00 | 0.060 | 0.511 | +0.939 | +0.137 |  |
| `register.tycker_vs_anser` | register | O | 1.00 | 2.00 | 0.037 | 0.511 | +0.730 | +0.119 |  |
| `register.valdigt_vs_mycket` | register | O | 1.00 | 1.50 | 0.036 | 0.510 | +0.854 | +0.134 |  |
| `register.sjalvklart` | register | O | 1.00 | 1.50 | 1.000 | 0.583 | +1.029 | +0.086 |  |
| `register.dem_har_rate` | register | T | 0.50 | 1.00 | 1.000 | 0.430 | +2.131 | +0.175 | x2.1; alone AUC<0.5 |
| `register.bland_annat` | register | O | 0.60 | 1.50 | 1.000 | 0.587 | +1.546 | +0.179 |  |
| `register.samt` | register | O | 1.00 | 1.50 | 1.000 | 0.587 | +1.124 | +0.137 |  |
| `register.bra` | register | O | 1.00 | 1.50 | 1.000 | 0.585 | +0.779 | +0.107 |  |
| `stats.long_words` | stats | T | 0.50 | 1.00 | 0.020 | 0.500 | +0.767 | +0.009 | alone AUC<0.5 |
| `stats.av_nominal` | stats | T | 1.00 | 1.00 | 0.020 | 0.492 | +0.690 | +0.020 | alone AUC<0.5 |
| `stats.av_other` | stats | T | 1.00 | 1.50 | 0.020 | 0.503 | +0.432 | +0.017 | x0.4 |
| `wordorder.fronted_theme` | wordorder | O | 0.80 | 3.00 | 0.139 | 0.531 | +1.055 | +0.185 |  |
| `wordorder.subject_theme` | wordorder | T | 0.80 | 2.00 | 0.532 | 0.523 | +1.148 | +0.095 |  |
| `wordorder.v2_violation` | wordorder | T | 1.00 | 3.00 | 0.001 | 0.501 | +1.364 | +0.050 |  |

Learned intercept (prior, log-odds): -0.047 (engine prior +0.000).

