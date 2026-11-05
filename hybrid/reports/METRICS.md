# Hybrid models on val

All models fitted on **all of trn** (sessions ≤ 2004-06-30; hyper-parameters by session-grouped 5-fold CV inside trn) and scored **once** on **val** (sessions 2004-07 .. 2006, balanced files) through the harness (`hybrid/eval_val.py`, results JSON with split="val", branch hybrid-ml). `tsw.engine:Default` and MeanSentLen are scored on the same records. `masked` = topic tokens replaced by XTOPIC before scoring; models are always fitted on unmasked trn text. Numbers are acc / macro-F1 / AUC. `trn CV AUC` = out-of-fold AUC inside trn for the chosen C (LR models).


## gran = sent (n = 21728)

| model | what | unmasked acc / F1 / AUC | masked acc / F1 / AUC | trn CV AUC |
|---|---|---|---|---:|
| `MeanSentLen` | baseline: mean sentence length (hand-set threshold) | 0.563 / 0.560 / **0.584** | 0.563 / 0.559 / **0.583** | - |
| `engine_raw` | pure rule engine `tsw.engine:Default` | 0.560 / 0.559 / **0.577** | 0.560 / 0.559 / **0.577** | - |
| `engine_calib` | engine score, LR-calibrated (threshold/scale only) | 0.562 / 0.560 / **0.577** | 0.562 / 0.560 / **0.577** | 0.586 |
| `stack_rules` | LR re-weighting of the engine's per-rule contributions | 0.562 / 0.559 / **0.578** | 0.562 / 0.558 / **0.578** | 0.590 |
| `stack_families` | LR over per-family subscores | 0.562 / 0.560 / **0.577** | 0.562 / 0.560 / **0.577** | 0.586 |
| `lr_engine` | LR over rule contributions + rule statistics | 0.576 / 0.576 / **0.608** | 0.576 / 0.576 / **0.607** | 0.622 |
| `gbt_engine` | GBT over rule contributions + rule statistics | 0.577 / 0.575 / **0.613** | 0.577 / 0.575 / **0.613** | - |
| `tree_engine` | depth-3 tree over rule statistics + engine score | 0.548 / 0.493 / **0.575** | 0.548 / 0.493 / **0.575** | - |
| `hybrid_plus` | engine score + 10 strongest literature features (LR) | 0.602 / 0.602 / **0.637** | 0.602 / 0.602 / **0.637** | 0.641 |
| `lr_expert` | LR over ~110 literature word-list features (Phase A) | 0.619 / 0.619 / **0.668** | 0.622 / 0.622 / **0.670** | 0.709 |
| `gbt` | GBT over the literature features (Phase A) | 0.628 / 0.628 / **0.682** | 0.629 / 0.629 / **0.684** | - |
| `tree_d3` | depth-3 tree over the literature features | 0.575 / 0.536 / **0.599** | 0.575 / 0.536 / **0.599** | - |
| `tree_d4` | depth-4 tree over the literature features | 0.580 / 0.544 / **0.604** | 0.580 / 0.544 / **0.604** | - |
| `lr_all` | LR over engine contributions + statistics + literature features | 0.622 / 0.622 / **0.668** | 0.624 / 0.624 / **0.671** | 0.710 |
| `gbt_all` | GBT over engine contributions + statistics + literature features | 0.631 / 0.631 / **0.683** | 0.632 / 0.632 / **0.684** | - |
| `bow_lr` | reference only: LR on the 300 most frequent words (no expert input) | 0.653 / 0.653 / **0.707** | 0.648 / 0.647 / **0.700** | 0.746 |

## gran = 100 (n = 3354)

| model | what | unmasked acc / F1 / AUC | masked acc / F1 / AUC | trn CV AUC |
|---|---|---|---|---:|
| `MeanSentLen` | baseline: mean sentence length (hand-set threshold) | 0.633 / 0.629 / **0.671** | 0.633 / 0.629 / **0.671** | - |
| `engine_raw` | pure rule engine `tsw.engine:Default` | 0.611 / 0.607 / **0.640** | 0.611 / 0.607 / **0.640** | - |
| `engine_calib` | engine score, LR-calibrated (threshold/scale only) | 0.614 / 0.613 / **0.640** | 0.614 / 0.613 / **0.640** | 0.710 |
| `stack_rules` | LR re-weighting of the engine's per-rule contributions | 0.617 / 0.615 / **0.635** | 0.617 / 0.615 / **0.635** | 0.717 |
| `stack_families` | LR over per-family subscores | 0.614 / 0.613 / **0.640** | 0.614 / 0.613 / **0.640** | 0.710 |
| `lr_engine` | LR over rule contributions + rule statistics | 0.677 / 0.676 / **0.740** | 0.677 / 0.676 / **0.740** | 0.763 |
| `gbt_engine` | GBT over rule contributions + rule statistics | 0.683 / 0.683 / **0.748** | 0.682 / 0.682 / **0.748** | - |
| `tree_engine` | depth-3 tree over rule statistics + engine score | 0.652 / 0.652 / **0.724** | 0.652 / 0.652 / **0.724** | - |
| `hybrid_plus` | engine score + 10 strongest literature features (LR) | 0.706 / 0.705 / **0.774** | 0.706 / 0.705 / **0.774** | 0.805 |
| `lr_expert` | LR over ~110 literature word-list features (Phase A) | 0.739 / 0.739 / **0.821** | 0.744 / 0.744 / **0.824** | 0.891 |
| `gbt` | GBT over the literature features (Phase A) | 0.752 / 0.752 / **0.836** | 0.754 / 0.754 / **0.838** | - |
| `tree_d3` | depth-3 tree over the literature features | 0.635 / 0.624 / **0.703** | 0.635 / 0.625 / **0.703** | - |
| `tree_d4` | depth-4 tree over the literature features | 0.671 / 0.666 / **0.728** | 0.671 / 0.666 / **0.728** | - |
| `lr_all` | LR over engine contributions + statistics + literature features | 0.737 / 0.737 / **0.815** | 0.741 / 0.741 / **0.818** | 0.893 |
| `gbt_all` | GBT over engine contributions + statistics + literature features | 0.758 / 0.758 / **0.840** | 0.758 / 0.758 / **0.842** | - |
| `bow_lr` | reference only: LR on the 300 most frequent words (no expert input) | 0.805 / 0.805 / **0.880** | 0.795 / 0.795 / **0.870** | 0.932 |

## gran = 500 (n = 636)

| model | what | unmasked acc / F1 / AUC | masked acc / F1 / AUC | trn CV AUC |
|---|---|---|---|---:|
| `MeanSentLen` | baseline: mean sentence length (hand-set threshold) | 0.709 / 0.702 / **0.793** | 0.708 / 0.700 / **0.792** | - |
| `engine_raw` | pure rule engine `tsw.engine:Default` | 0.700 / 0.695 / **0.791** | 0.700 / 0.695 / **0.790** | - |
| `engine_calib` | engine score, LR-calibrated (threshold/scale only) | 0.714 / 0.713 / **0.791** | 0.714 / 0.713 / **0.790** | 0.846 |
| `stack_rules` | LR re-weighting of the engine's per-rule contributions | 0.678 / 0.672 / **0.755** | 0.678 / 0.672 / **0.755** | 0.853 |
| `stack_families` | LR over per-family subscores | 0.714 / 0.713 / **0.791** | 0.714 / 0.713 / **0.790** | 0.846 |
| `lr_engine` | LR over rule contributions + rule statistics | 0.800 / 0.799 / **0.882** | 0.800 / 0.799 / **0.882** | 0.902 |
| `gbt_engine` | GBT over rule contributions + rule statistics | 0.811 / 0.811 / **0.889** | 0.811 / 0.811 / **0.889** | - |
| `tree_engine` | depth-3 tree over rule statistics + engine score | 0.811 / 0.810 / **0.855** | 0.811 / 0.810 / **0.855** | - |
| `hybrid_plus` | engine score + 10 strongest literature features (LR) | 0.832 / 0.831 / **0.925** | 0.830 / 0.829 / **0.925** | 0.946 |
| `lr_expert` | LR over ~110 literature word-list features (Phase A) | 0.898 / 0.898 / **0.962** | 0.896 / 0.896 / **0.964** | 0.984 |
| `gbt` | GBT over the literature features (Phase A) | 0.892 / 0.892 / **0.957** | 0.901 / 0.901 / **0.959** | - |
| `tree_d3` | depth-3 tree over the literature features | 0.722 / 0.717 / **0.818** | 0.722 / 0.717 / **0.818** | - |
| `tree_d4` | depth-4 tree over the literature features | 0.748 / 0.745 / **0.827** | 0.748 / 0.745 / **0.826** | - |
| `lr_all` | LR over engine contributions + statistics + literature features | 0.881 / 0.880 / **0.947** | 0.879 / 0.879 / **0.949** | 0.985 |
| `gbt_all` | GBT over engine contributions + statistics + literature features | 0.890 / 0.890 / **0.959** | 0.890 / 0.890 / **0.960** | - |
| `bow_lr` | reference only: LR on the 300 most frequent words (no expert input) | 0.914 / 0.914 / **0.975** | 0.909 / 0.909 / **0.971** | 0.995 |

Engine scored at commit `e1dd1e9b`.
