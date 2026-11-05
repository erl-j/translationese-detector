# Hybrid models on dev-val

All models fitted on **dev-train** (dev sessions before 2002-03-01) and evaluated on **dev-val** (sessions 2002-03-01 .. 2004-05-05, balanced files; see hybrid/SPLIT.md). The pure engine and MeanSentLen are scored on exactly the same records. `masked` = topic tokens replaced by XTOPIC before scoring (evals/topicmask.py); models were always fitted on unmasked dev-train text. `unseen spk` = AUC on dev-val units none of whose speakers occur in dev-train (sent n=2139, 100 n=256; too few at 500). Numbers are acc / macro-F1 / AUC.

Engine: `tsw.engine:Default`, tsw source hash `ae5041cbd85649bf`.


## gran = sent (n = 29183)

| model | what | unmasked acc / F1 / AUC | masked acc / F1 / AUC | unseen spk AUC |
|---|---|---|---|---:|
| `MeanSentLen` | baseline: mean sentence length (hand-set threshold) | 0.567 / 0.567 / **0.601** | 0.567 / 0.567 / **0.601** | - |
| `engine_raw` | pure rule engine `tsw.engine:Default` | 0.564 / 0.564 / **0.597** | 0.564 / 0.564 / **0.597** | 0.575 |
| `engine_calib` | engine score, LR-calibrated (threshold/scale only) | 0.546 / 0.529 / **0.597** | 0.545 / 0.529 / **0.597** | 0.575 |
| `stack_rules` | LR re-weighting of the engine's per-rule contributions | 0.565 / 0.559 / **0.619** | 0.565 / 0.559 / **0.619** | 0.607 |
| `stack_families` | LR over per-family subscores | 0.546 / 0.529 / **0.597** | 0.545 / 0.529 / **0.597** | 0.575 |
| `lr_engine` | LR over rule contributions + rule statistics | 0.575 / 0.572 / **0.631** | 0.575 / 0.572 / **0.631** | 0.616 |
| `gbt_engine` | GBT over rule contributions + rule statistics | 0.563 / 0.552 / **0.630** | 0.563 / 0.552 / **0.630** | 0.623 |
| `tree_engine` | depth-3 tree over rule statistics + engine score | 0.528 / 0.486 / **0.587** | 0.528 / 0.486 / **0.586** | 0.592 |
| `hybrid_plus` | engine score + 10 strongest literature features (LR) | 0.589 / 0.588 / **0.642** | 0.588 / 0.587 / **0.641** | 0.653 |
| `lr_expert` | LR over ~110 literature word-list features (Phase A) | 0.632 / 0.632 / **0.697** | 0.632 / 0.632 / **0.697** | 0.704 |
| `gbt` | GBT over the literature features (Phase A) | 0.637 / 0.637 / **0.705** | 0.637 / 0.637 / **0.705** | 0.706 |
| `tree_d3` | depth-3 tree over the literature features | 0.539 / 0.511 / **0.583** | 0.539 / 0.510 / **0.583** | 0.594 |
| `tree_d4` | depth-4 tree over the literature features | 0.544 / 0.519 / **0.587** | 0.543 / 0.519 / **0.586** | 0.600 |
| `lr_all` | LR over engine contributions + statistics + literature features | 0.631 / 0.631 / **0.698** | 0.631 / 0.631 / **0.698** | 0.705 |
| `gbt_all` | GBT over engine contributions + statistics + literature features | 0.639 / 0.639 / **0.705** | 0.639 / 0.639 / **0.705** | 0.704 |
| `bow_lr` | reference only: LR on the 300 most frequent words (no expert input) | 0.663 / 0.661 / **0.721** | 0.659 / 0.657 / **0.714** | 0.743 |

## gran = 100 (n = 4610)

| model | what | unmasked acc / F1 / AUC | masked acc / F1 / AUC | unseen spk AUC |
|---|---|---|---|---:|
| `MeanSentLen` | baseline: mean sentence length (hand-set threshold) | 0.655 / 0.654 / **0.720** | 0.655 / 0.654 / **0.719** | - |
| `engine_raw` | pure rule engine `tsw.engine:Default` | 0.670 / 0.661 / **0.727** | 0.670 / 0.661 / **0.726** | 0.678 |
| `engine_calib` | engine score, LR-calibrated (threshold/scale only) | 0.662 / 0.661 / **0.727** | 0.661 / 0.661 / **0.726** | 0.678 |
| `stack_rules` | LR re-weighting of the engine's per-rule contributions | 0.696 / 0.695 / **0.761** | 0.695 / 0.694 / **0.761** | 0.741 |
| `stack_families` | LR over per-family subscores | 0.662 / 0.661 / **0.727** | 0.661 / 0.661 / **0.726** | 0.678 |
| `lr_engine` | LR over rule contributions + rule statistics | 0.703 / 0.702 / **0.774** | 0.703 / 0.702 / **0.773** | 0.785 |
| `gbt_engine` | GBT over rule contributions + rule statistics | 0.705 / 0.704 / **0.771** | 0.705 / 0.704 / **0.771** | 0.781 |
| `tree_engine` | depth-3 tree over rule statistics + engine score | 0.669 / 0.669 / **0.735** | 0.669 / 0.669 / **0.734** | 0.730 |
| `hybrid_plus` | engine score + 10 strongest literature features (LR) | 0.729 / 0.726 / **0.801** | 0.727 / 0.724 / **0.801** | 0.805 |
| `lr_expert` | LR over ~110 literature word-list features (Phase A) | 0.802 / 0.800 / **0.881** | 0.800 / 0.799 / **0.880** | 0.895 |
| `gbt` | GBT over the literature features (Phase A) | 0.796 / 0.795 / **0.877** | 0.796 / 0.795 / **0.877** | 0.888 |
| `tree_d3` | depth-3 tree over the literature features | 0.638 / 0.632 / **0.705** | 0.638 / 0.631 / **0.705** | 0.746 |
| `tree_d4` | depth-4 tree over the literature features | 0.666 / 0.626 / **0.731** | 0.666 / 0.626 / **0.731** | 0.744 |
| `lr_all` | LR over engine contributions + statistics + literature features | 0.803 / 0.801 / **0.882** | 0.802 / 0.800 / **0.882** | 0.889 |
| `gbt_all` | GBT over engine contributions + statistics + literature features | 0.800 / 0.798 / **0.880** | 0.800 / 0.799 / **0.879** | 0.889 |
| `bow_lr` | reference only: LR on the 300 most frequent words (no expert input) | 0.830 / 0.828 / **0.909** | 0.819 / 0.817 / **0.900** | 0.932 |

## gran = 500 (n = 911)

| model | what | unmasked acc / F1 / AUC | masked acc / F1 / AUC | unseen spk AUC |
|---|---|---|---|---:|
| `MeanSentLen` | baseline: mean sentence length (hand-set threshold) | 0.740 / 0.740 / **0.846** | 0.740 / 0.740 / **0.845** | - |
| `engine_raw` | pure rule engine `tsw.engine:Default` | 0.788 / 0.779 / **0.854** | 0.788 / 0.779 / **0.854** | - |
| `engine_calib` | engine score, LR-calibrated (threshold/scale only) | 0.768 / 0.764 / **0.854** | 0.768 / 0.764 / **0.854** | - |
| `stack_rules` | LR re-weighting of the engine's per-rule contributions | 0.777 / 0.773 / **0.859** | 0.776 / 0.772 / **0.858** | - |
| `stack_families` | LR over per-family subscores | 0.768 / 0.764 / **0.854** | 0.768 / 0.764 / **0.854** | - |
| `lr_engine` | LR over rule contributions + rule statistics | 0.838 / 0.836 / **0.907** | 0.838 / 0.836 / **0.906** | - |
| `gbt_engine` | GBT over rule contributions + rule statistics | 0.832 / 0.829 / **0.897** | 0.830 / 0.827 / **0.896** | - |
| `tree_engine` | depth-3 tree over rule statistics + engine score | 0.788 / 0.784 / **0.857** | 0.787 / 0.783 / **0.856** | - |
| `hybrid_plus` | engine score + 10 strongest literature features (LR) | 0.864 / 0.861 / **0.934** | 0.862 / 0.859 / **0.934** | - |
| `lr_expert` | LR over ~110 literature word-list features (Phase A) | 0.940 / 0.939 / **0.982** | 0.939 / 0.938 / **0.982** | - |
| `gbt` | GBT over the literature features (Phase A) | 0.909 / 0.908 / **0.970** | 0.909 / 0.908 / **0.970** | - |
| `tree_d3` | depth-3 tree over the literature features | 0.767 / 0.758 / **0.816** | 0.768 / 0.759 / **0.815** | - |
| `tree_d4` | depth-4 tree over the literature features | 0.789 / 0.781 / **0.849** | 0.790 / 0.782 / **0.849** | - |
| `lr_all` | LR over engine contributions + statistics + literature features | 0.935 / 0.934 / **0.982** | 0.933 / 0.932 / **0.982** | - |
| `gbt_all` | GBT over engine contributions + statistics + literature features | 0.908 / 0.907 / **0.969** | 0.908 / 0.907 / **0.969** | - |
| `bow_lr` | reference only: LR on the 300 most frequent words (no expert input) | 0.957 / 0.956 / **0.993** | 0.953 / 0.952 / **0.991** | - |
