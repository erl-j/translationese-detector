# Learned vs hand-set rule weights (gran=500)

LR over the engine's per-rule contributions on trn (2734 units; C=0.1 chosen by session-grouped CV inside trn). `multiplier` is the learned coefficient on the hand-set contribution: 1.0 means the hand-set weight is about right, >1 under-weighted, between 0 and 1 over-weighted, **< 0 means the sign flips** (the rule points the wrong way once the other rules are accounted for). `alone AUC` is the rule's contribution used alone as a score on trn; `fires` is the share of units with a non-zero contribution. Engine: `tsw.engine:Default`, tsw hash `00a590c8e3449552`, commit `cef0b257`.

| rule | family | dir | hand weight | cap | fires | alone AUC | multiplier | std. coef | flag |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `calques.lexicon` | calques | T | 1.00 | 3.00 | 0.500 | 0.795 | +1.002 | +1.008 |  |
| `calques.native_counterparts` | calques | O | 1.00 | 2.00 | 0.031 | 0.524 | +0.627 | +0.157 |  |
| `calques.forsoka_att` | calques | T | 1.00 | 1.50 | 0.208 | 0.526 | +0.953 | +0.239 |  |
| `calques.pied_piped_relative` | calques | T | 1.00 | 1.50 | 0.221 | 0.578 | +0.141 | +0.057 | x0.1 |
| `connpunct.native_connectives` | connpunct | O | 0.50 | 2.00 | 1.000 | 0.771 | +1.126 | +0.790 |  |
| `connpunct.additive_ocksa` | connpunct | O | 0.40 | 1.50 | 1.000 | 0.765 | +1.484 | +0.920 |  |
| `connpunct.calque_connectives` | connpunct | T | 0.80 | 2.00 | 1.000 | 0.656 | +0.516 | +0.411 |  |
| `connpunct.att_drop` | connpunct | O | 1.00 | 1.50 | 1.000 | 0.558 | +0.979 | +0.197 |  |
| `connpunct.som_drop` | connpunct | T | 0.70 | 1.50 | 0.354 | 0.549 | +0.450 | +0.104 | x0.5 |
| `connpunct.initial_och` | connpunct | T | 1.00 | 2.00 | 1.000 | 0.500 | +0.770 | +0.296 |  |
| `connpunct.initial_men` | connpunct | T | 0.70 | 1.50 | 1.000 | 0.554 | +2.060 | +0.535 | x2.1 |
| `connpunct.initial_adverbial_comma` | connpunct | T | 1.00 | 2.00 | 0.880 | 0.665 | +0.434 | +0.269 | x0.4 |
| `connpunct.clause_comma_omission` | connpunct | T | 0.70 | 1.50 | 0.718 | 0.606 | +0.536 | +0.207 |  |
| `connpunct.written_out_abbrev` | connpunct | T | 0.50 | 1.50 | 0.446 | 0.644 | +1.120 | +0.496 |  |
| `connpunct.parenthetical_dash` | connpunct | T | 0.50 | 1.50 | 1.000 | 0.618 | +0.710 | +0.357 |  |
| `connpunct.explanatory_parens` | connpunct | T | 1.00 | 1.00 | 1.000 | 0.509 | +0.191 | +0.055 | x0.2 |
| `connpunct.exclamation` | connpunct | O | 0.50 | 1.50 | 1.000 | 0.594 | +1.019 | +0.303 |  |
| `starter.modal_particles` | starter | O | 1.00 | 3.00 | 1.000 | 0.684 | +0.535 | +0.606 |  |
| `starter.sent_len` | starter | T | 0.50 | 2.00 | 0.996 | 0.781 | +0.735 | +0.731 |  |
| `particles.namligen` | particles | O | 1.00 | 2.50 | 1.000 | 0.614 | +0.573 | +0.445 |  |
| `particles.anda` | particles | O | 1.00 | 2.00 | 1.000 | 0.598 | +0.749 | +0.313 |  |
| `particles.alltsa` | particles | O | 1.00 | 1.50 | 1.000 | 0.587 | +0.519 | +0.207 |  |
| `particles.stance` | particles | O | 1.00 | 2.00 | 1.000 | 0.598 | +0.787 | +0.436 |  |
| `particles.da_anaphoric` | particles | O | 1.00 | 2.00 | 1.000 | 0.555 | +0.442 | +0.105 | x0.4 |
| `particles.of_course` | particles | T | 1.00 | 1.50 | 0.293 | 0.504 | +1.306 | +0.325 |  |
| `particles.dvs` | particles | O | 1.00 | 1.50 | 1.000 | 0.618 | +0.597 | +0.300 |  |
| `particles.particle_verbs` | particles | O | 1.00 | 1.50 | 1.000 | 0.587 | +0.135 | +0.057 | x0.1 |
| `particles.ta_itu_med` | particles | T | 1.00 | 2.00 | 1.000 | 0.542 | +0.458 | +0.260 | x0.5 |
| `particles.generic_man` | particles | O | 0.25 | 1.50 | 1.000 | 0.650 | +0.784 | +0.145 |  |
| `particles.copular_det` | particles | O | 0.50 | 1.50 | 1.000 | 0.641 | +0.624 | +0.191 |  |
| `particles.poss_1st` | particles | T | 0.60 | 1.50 | 1.000 | 0.614 | +0.798 | +0.409 |  |
| `particles.poss_3rd` | particles | T | 0.80 | 2.00 | 1.000 | 0.684 | +0.574 | +0.495 |  |
| `particles.han_hon` | particles | T | 0.50 | 1.00 | 1.000 | 0.597 | +0.859 | +0.331 |  |
| `particles.ni_vs_man` | particles | T | 1.00 | 1.50 | 0.755 | 0.641 | +0.259 | +0.182 | x0.3 |
| `register.kommer_att_vs_ska` | register | T | 1.00 | 2.50 | 0.950 | 0.804 | +0.347 | +0.570 | x0.3 |
| `register.skulle_vilja_vs_vill` | register | T | 1.00 | 2.00 | 0.738 | 0.659 | +0.649 | +0.498 |  |
| `register.tycker_vs_anser` | register | O | 1.00 | 2.00 | 0.534 | 0.645 | +0.062 | +0.054 | x0.1 |
| `register.valdigt_vs_mycket` | register | O | 1.00 | 1.50 | 0.553 | 0.630 | +0.594 | +0.411 |  |
| `register.sjalvklart` | register | O | 1.00 | 1.50 | 1.000 | 0.583 | +0.835 | +0.365 |  |
| `register.dem_har_rate` | register | T | 0.50 | 1.00 | 1.000 | 0.561 | +1.520 | +0.510 |  |
| `register.bland_annat` | register | O | 0.60 | 1.50 | 1.000 | 0.635 | +1.077 | +0.614 |  |
| `register.samt` | register | O | 1.00 | 1.50 | 1.000 | 0.607 | +0.972 | +0.544 |  |
| `register.bra` | register | O | 1.00 | 1.50 | 1.000 | 0.646 | +0.374 | +0.238 | x0.4 |
| `stats.long_words` | stats | T | 0.50 | 1.00 | 1.000 | 0.602 | +0.066 | +0.008 | x0.1 |
| `stats.av_nominal` | stats | T | 1.00 | 1.00 | 1.000 | 0.620 | +0.699 | +0.301 |  |
| `stats.av_other` | stats | T | 1.00 | 1.50 | 1.000 | 0.672 | +0.612 | +0.343 |  |
| `wordorder.fronted_theme` | wordorder | O | 0.80 | 3.00 | 0.960 | 0.781 | +0.642 | +0.665 |  |
| `wordorder.subject_theme` | wordorder | T | 0.80 | 2.00 | 1.000 | 0.575 | +0.424 | +0.200 | x0.4 |
| `wordorder.v2_violation` | wordorder | T | 1.00 | 3.00 | 0.015 | 0.512 | +1.003 | +0.206 |  |

Learned intercept (prior, log-odds): -0.674 (engine prior +0.000).

