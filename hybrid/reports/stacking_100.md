# Learned vs hand-set rule weights (gran=100)

LR over the engine's per-rule contributions on trn (14170 units; C=0.03 chosen by session-grouped CV inside trn). `multiplier` is the learned coefficient on the hand-set contribution: 1.0 means the hand-set weight is about right, >1 under-weighted, between 0 and 1 over-weighted, **< 0 means the sign flips** (the rule points the wrong way once the other rules are accounted for). `alone AUC` is the rule's contribution used alone as a score on trn; `fires` is the share of units with a non-zero contribution. Engine: `tsw.engine:Default`, tsw hash `00a590c8e3449552`, commit `cef0b257`.

| rule | family | dir | hand weight | cap | fires | alone AUC | multiplier | std. coef | flag |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `calques.lexicon` | calques | T | 1.00 | 3.00 | 0.165 | 0.603 | +1.460 | +0.706 |  |
| `calques.native_counterparts` | calques | O | 1.00 | 2.00 | 0.007 | 0.506 | +1.325 | +0.162 |  |
| `calques.forsoka_att` | calques | T | 1.00 | 1.50 | 0.053 | 0.505 | +0.933 | +0.112 |  |
| `calques.pied_piped_relative` | calques | T | 1.00 | 1.50 | 0.056 | 0.521 | +0.724 | +0.144 |  |
| `connpunct.native_connectives` | connpunct | O | 0.50 | 2.00 | 1.000 | 0.640 | +1.904 | +0.544 |  |
| `connpunct.additive_ocksa` | connpunct | O | 0.40 | 1.50 | 1.000 | 0.641 | +2.067 | +0.517 | x2.1 |
| `connpunct.calque_connectives` | connpunct | T | 0.80 | 2.00 | 1.000 | 0.520 | +0.866 | +0.323 |  |
| `connpunct.att_drop` | connpunct | O | 1.00 | 1.50 | 1.000 | 0.547 | +0.446 | +0.043 | x0.4 |
| `connpunct.som_drop` | connpunct | T | 0.70 | 1.50 | 0.096 | 0.517 | +1.068 | +0.114 |  |
| `connpunct.initial_och` | connpunct | T | 1.00 | 2.00 | 1.000 | 0.465 | +1.419 | +0.258 | alone AUC<0.5 |
| `connpunct.initial_men` | connpunct | T | 0.70 | 1.50 | 1.000 | 0.490 | +2.385 | +0.279 | x2.4; alone AUC<0.5 |
| `connpunct.initial_adverbial_comma` | connpunct | T | 1.00 | 2.00 | 0.410 | 0.569 | +0.678 | +0.209 |  |
| `connpunct.clause_comma_omission` | connpunct | T | 0.70 | 1.50 | 0.258 | 0.537 | +0.812 | +0.146 |  |
| `connpunct.written_out_abbrev` | connpunct | T | 0.50 | 1.50 | 0.128 | 0.547 | +1.529 | +0.280 |  |
| `connpunct.parenthetical_dash` | connpunct | T | 0.50 | 1.50 | 1.000 | 0.523 | +1.219 | +0.277 |  |
| `connpunct.explanatory_parens` | connpunct | T | 1.00 | 1.00 | 1.000 | 0.469 | +0.535 | +0.073 | alone AUC<0.5 |
| `connpunct.exclamation` | connpunct | O | 0.50 | 1.50 | 1.000 | 0.557 | +1.531 | +0.213 |  |
| `starter.modal_particles` | starter | O | 1.00 | 3.00 | 1.000 | 0.591 | +0.802 | +0.477 |  |
| `starter.sent_len` | starter | T | 0.50 | 2.00 | 0.992 | 0.675 | +1.379 | +0.514 |  |
| `particles.namligen` | particles | O | 1.00 | 2.50 | 1.000 | 0.563 | +1.287 | +0.490 |  |
| `particles.anda` | particles | O | 1.00 | 2.00 | 1.000 | 0.558 | +0.738 | +0.143 |  |
| `particles.alltsa` | particles | O | 1.00 | 1.50 | 1.000 | 0.557 | +0.810 | +0.161 |  |
| `particles.stance` | particles | O | 1.00 | 2.00 | 1.000 | 0.560 | +0.854 | +0.230 |  |
| `particles.da_anaphoric` | particles | O | 1.00 | 2.00 | 1.000 | 0.547 | +0.895 | +0.100 |  |
| `particles.of_course` | particles | T | 1.00 | 1.50 | 0.080 | 0.501 | +1.369 | +0.146 |  |
| `particles.dvs` | particles | O | 1.00 | 1.50 | 1.000 | 0.565 | +0.908 | +0.236 |  |
| `particles.particle_verbs` | particles | O | 1.00 | 1.50 | 1.000 | 0.555 | +0.731 | +0.152 |  |
| `particles.ta_itu_med` | particles | T | 1.00 | 2.00 | 1.000 | 0.477 | +1.069 | +0.302 | alone AUC<0.5 |
| `particles.generic_man` | particles | O | 0.25 | 1.50 | 1.000 | 0.592 | +1.293 | +0.093 |  |
| `particles.copular_det` | particles | O | 0.50 | 1.50 | 1.000 | 0.572 | +0.624 | +0.084 |  |
| `particles.poss_1st` | particles | T | 0.60 | 1.50 | 1.000 | 0.543 | +1.270 | +0.277 |  |
| `particles.poss_3rd` | particles | T | 0.80 | 2.00 | 1.000 | 0.538 | +0.736 | +0.296 |  |
| `particles.han_hon` | particles | T | 0.50 | 1.00 | 1.000 | 0.507 | +1.372 | +0.260 |  |
| `particles.ni_vs_man` | particles | T | 1.00 | 1.50 | 0.312 | 0.574 | +0.455 | +0.152 | x0.5 |
| `register.kommer_att_vs_ska` | register | T | 1.00 | 2.50 | 0.524 | 0.667 | +0.713 | +0.575 |  |
| `register.skulle_vilja_vs_vill` | register | T | 1.00 | 2.00 | 0.270 | 0.559 | +0.838 | +0.292 |  |
| `register.tycker_vs_anser` | register | O | 1.00 | 2.00 | 0.177 | 0.550 | +0.395 | +0.163 | x0.4 |
| `register.valdigt_vs_mycket` | register | O | 1.00 | 1.50 | 0.177 | 0.539 | +0.603 | +0.220 |  |
| `register.sjalvklart` | register | O | 1.00 | 1.50 | 1.000 | 0.555 | +0.782 | +0.160 |  |
| `register.dem_har_rate` | register | T | 0.50 | 1.00 | 1.000 | 0.492 | +2.017 | +0.359 | x2.0; alone AUC<0.5 |
| `register.bland_annat` | register | O | 0.60 | 1.50 | 1.000 | 0.568 | +1.408 | +0.399 |  |
| `register.samt` | register | O | 1.00 | 1.50 | 1.000 | 0.567 | +1.244 | +0.354 |  |
| `register.bra` | register | O | 1.00 | 1.50 | 1.000 | 0.572 | +0.579 | +0.186 |  |
| `stats.long_words` | stats | T | 0.50 | 1.00 | 1.000 | 0.552 | +0.969 | +0.094 |  |
| `stats.av_nominal` | stats | T | 1.00 | 1.00 | 1.000 | 0.511 | +0.751 | +0.187 |  |
| `stats.av_other` | stats | T | 1.00 | 1.50 | 1.000 | 0.594 | +0.707 | +0.236 |  |
| `wordorder.fronted_theme` | wordorder | O | 0.80 | 3.00 | 0.547 | 0.637 | +0.954 | +0.423 |  |
| `wordorder.subject_theme` | wordorder | T | 0.80 | 2.00 | 0.961 | 0.532 | +0.748 | +0.153 |  |
| `wordorder.v2_violation` | wordorder | T | 1.00 | 3.00 | 0.003 | 0.502 | +0.922 | +0.075 |  |

Learned intercept (prior, log-odds): -0.174 (engine prior +0.000).

