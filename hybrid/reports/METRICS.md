# Hybrid models on val

All models fitted on **all of trn** (sessions ≤ 2004-06-30; hyper-parameters by session-grouped 5-fold CV inside trn) and scored **once** on **val** (sessions 2004-07 .. 2006, balanced files) through the harness (`hybrid/eval_val.py`, results JSON with split="val", branch hybrid-ml). `tsw.engine:Default` and MeanSentLen are scored on the same records. `masked` = topic tokens replaced by XTOPIC before scoring; models are always fitted on unmasked trn text. Numbers are acc / macro-F1 / AUC. `trn CV AUC` = out-of-fold AUC inside trn for the chosen C (LR models).


## gran = sent (n = 21728)

| model | what | unmasked acc / F1 / AUC | masked acc / F1 / AUC | trn CV AUC |
|---|---|---|---|---:|
| `MeanSentLen` | baseline: mean sentence length (hand-set threshold) | 0.563 / 0.560 / **0.584** | 0.563 / 0.559 / **0.583** | - |
| `engine_raw` | pure rule engine `tsw.engine:Default` | 0.612 / 0.612 / **0.666** | 0.612 / 0.612 / **0.666** | - |
| `engine_calib` | engine score, LR-calibrated (threshold/scale only) | 0.616 / 0.615 / **0.666** | 0.616 / 0.615 / **0.666** | 0.723 |
| `stack_rules` | LR re-weighting of the engine's per-rule contributions | 0.618 / 0.617 / **0.670** | 0.618 / 0.617 / **0.669** | 0.729 |
| `stack_families` | LR over per-family subscores | 0.617 / 0.615 / **0.667** | 0.617 / 0.615 / **0.667** | 0.726 |
| `lr_engine` | LR over rule contributions + rule statistics | 0.621 / 0.621 / **0.675** | 0.621 / 0.621 / **0.675** | 0.735 |
| `gbt_engine` | GBT over rule contributions + rule statistics | 0.626 / 0.625 / **0.681** | 0.625 / 0.625 / **0.681** | - |
| `tree_engine` | depth-3 tree over rule statistics + engine score | 0.616 / 0.615 / **0.663** | 0.616 / 0.614 / **0.663** | - |
| `hybrid_plus` | engine score + 10 strongest literature features (LR) | 0.630 / 0.629 / **0.681** | 0.629 / 0.629 / **0.681** | 0.730 |
| `lr_expert` | LR over ~110 literature word-list features (Phase A) | 0.619 / 0.619 / **0.668** | 0.622 / 0.622 / **0.670** | 0.709 |
| `gbt` | GBT over the literature features (Phase A) | 0.628 / 0.628 / **0.682** | 0.629 / 0.629 / **0.684** | - |
| `tree_d3` | depth-3 tree over the literature features | 0.575 / 0.536 / **0.599** | 0.575 / 0.536 / **0.599** | - |
| `tree_d4` | depth-4 tree over the literature features | 0.580 / 0.544 / **0.604** | 0.580 / 0.544 / **0.604** | - |
| `lr_all` | LR over engine contributions + statistics + literature features | 0.632 / 0.632 / **0.687** | 0.633 / 0.633 / **0.688** | 0.753 |
| `gbt_all` | GBT over engine contributions + statistics + literature features | 0.640 / 0.640 / **0.699** | 0.642 / 0.642 / **0.701** | - |
| `bow_lr` | reference only: LR on the 300 most frequent words (no expert input) | 0.653 / 0.653 / **0.707** | 0.648 / 0.647 / **0.700** | 0.746 |

## gran = 100 (n = 3354)

| model | what | unmasked acc / F1 / AUC | masked acc / F1 / AUC | trn CV AUC |
|---|---|---|---|---:|
| `MeanSentLen` | baseline: mean sentence length (hand-set threshold) | 0.633 / 0.629 / **0.671** | 0.633 / 0.629 / **0.671** | - |
| `engine_raw` | pure rule engine `tsw.engine:Default` | 0.747 / 0.746 / **0.833** | 0.747 / 0.746 / **0.833** | - |
| `engine_calib` | engine score, LR-calibrated (threshold/scale only) | 0.744 / 0.744 / **0.833** | 0.744 / 0.744 / **0.833** | 0.907 |
| `stack_rules` | LR re-weighting of the engine's per-rule contributions | 0.755 / 0.755 / **0.840** | 0.755 / 0.755 / **0.840** | 0.919 |
| `stack_families` | LR over per-family subscores | 0.740 / 0.740 / **0.830** | 0.741 / 0.741 / **0.830** | 0.911 |
| `lr_engine` | LR over rule contributions + rule statistics | 0.760 / 0.760 / **0.847** | 0.759 / 0.759 / **0.847** | 0.922 |
| `gbt_engine` | GBT over rule contributions + rule statistics | 0.758 / 0.758 / **0.845** | 0.758 / 0.758 / **0.845** | - |
| `tree_engine` | depth-3 tree over rule statistics + engine score | 0.752 / 0.752 / **0.827** | 0.752 / 0.752 / **0.828** | - |
| `hybrid_plus` | engine score + 10 strongest literature features (LR) | 0.760 / 0.760 / **0.850** | 0.761 / 0.761 / **0.850** | 0.916 |
| `lr_expert` | LR over ~110 literature word-list features (Phase A) | 0.739 / 0.739 / **0.821** | 0.744 / 0.744 / **0.824** | 0.891 |
| `gbt` | GBT over the literature features (Phase A) | 0.752 / 0.752 / **0.836** | 0.754 / 0.754 / **0.838** | - |
| `tree_d3` | depth-3 tree over the literature features | 0.635 / 0.624 / **0.703** | 0.635 / 0.625 / **0.703** | - |
| `tree_d4` | depth-4 tree over the literature features | 0.671 / 0.666 / **0.728** | 0.671 / 0.666 / **0.728** | - |
| `lr_all` | LR over engine contributions + statistics + literature features | 0.774 / 0.774 / **0.858** | 0.776 / 0.776 / **0.860** | 0.936 |
| `gbt_all` | GBT over engine contributions + statistics + literature features | 0.786 / 0.786 / **0.868** | 0.787 / 0.787 / **0.869** | - |
| `bow_lr` | reference only: LR on the 300 most frequent words (no expert input) | 0.805 / 0.805 / **0.880** | 0.795 / 0.795 / **0.870** | 0.932 |

## gran = 500 (n = 636)

| model | what | unmasked acc / F1 / AUC | masked acc / F1 / AUC | trn CV AUC |
|---|---|---|---|---:|
| `MeanSentLen` | baseline: mean sentence length (hand-set threshold) | 0.709 / 0.702 / **0.793** | 0.708 / 0.700 / **0.792** | - |
| `engine_raw` | pure rule engine `tsw.engine:Default` | 0.920 / 0.920 / **0.977** | 0.920 / 0.920 / **0.977** | - |
| `engine_calib` | engine score, LR-calibrated (threshold/scale only) | 0.931 / 0.931 / **0.977** | 0.931 / 0.931 / **0.977** | 0.990 |
| `stack_rules` | LR re-weighting of the engine's per-rule contributions | 0.912 / 0.912 / **0.972** | 0.912 / 0.912 / **0.972** | 0.993 |
| `stack_families` | LR over per-family subscores | 0.926 / 0.926 / **0.974** | 0.926 / 0.926 / **0.974** | 0.991 |
| `lr_engine` | LR over rule contributions + rule statistics | 0.925 / 0.925 / **0.972** | 0.923 / 0.923 / **0.972** | 0.992 |
| `gbt_engine` | GBT over rule contributions + rule statistics | 0.901 / 0.901 / **0.968** | 0.899 / 0.899 / **0.968** | - |
| `tree_engine` | depth-3 tree over rule statistics + engine score | 0.909 / 0.909 / **0.969** | 0.909 / 0.909 / **0.969** | - |
| `hybrid_plus` | engine score + 10 strongest literature features (LR) | 0.928 / 0.928 / **0.982** | 0.928 / 0.928 / **0.981** | 0.992 |
| `lr_expert` | LR over ~110 literature word-list features (Phase A) | 0.898 / 0.898 / **0.962** | 0.896 / 0.896 / **0.964** | 0.984 |
| `gbt` | GBT over the literature features (Phase A) | 0.892 / 0.892 / **0.957** | 0.901 / 0.901 / **0.959** | - |
| `tree_d3` | depth-3 tree over the literature features | 0.722 / 0.717 / **0.818** | 0.722 / 0.717 / **0.818** | - |
| `tree_d4` | depth-4 tree over the literature features | 0.748 / 0.745 / **0.827** | 0.748 / 0.745 / **0.826** | - |
| `lr_all` | LR over engine contributions + statistics + literature features | 0.939 / 0.939 / **0.982** | 0.939 / 0.939 / **0.983** | 0.994 |
| `gbt_all` | GBT over engine contributions + statistics + literature features | 0.910 / 0.910 / **0.971** | 0.906 / 0.906 / **0.971** | - |
| `bow_lr` | reference only: LR on the 300 most frequent words (no expert input) | 0.914 / 0.914 / **0.975** | 0.909 / 0.909 / **0.971** | 0.995 |

Engine scored at commit `cef0b257`.
