# What the expert-feature LR learned (`lr_expert`, fitted on all of trn)

Per granularity, the 25 features with the largest standardised coefficient (effect of +1 SD of the feature on the log-odds of *translated*). `lit` = predicted direction in docs/literature.md (T↑ more frequent in translations, T↓ less, ? = calibrate); `data` = direction of the class means on trn, with the feature's AUC alone (> .5 = higher in T); `LR` = sign of the coefficient given all other features. ✗ marks disagreement with the literature. LR signs of collinear features (e.g. a rate and its ratio) can differ from their univariate direction; trust `data` for the direction of a single cue, `LR` for its unique contribution.


## gran = sent (C = 0.01)

| # | feature | plain meaning | lit | data (AUC) | LR std. coef | |
|---:|---|---|---|---|---:|---|
| 1 | `mean_sent_len` | mean sentence length (words) (E1) | T↑ | T↑ (0.581) | +0.443 |  |
| 2 | `rate_p_ju` | modal particle ju per 1000 words (A5) | T↓ | T↓ (0.492) | -0.360 |  |
| 3 | `rate_p_namligen` | nämligen per 1000 words (A5) | T↓ | T↓ (0.493) | -0.246 |  |
| 4 | `r_skall` | share skall among skall+ska (A2) | T↑ | T↓ (0.473) | -0.219 | ✗ data ✗ LR |
| 5 | `rate_dem_O` | den/det/de här\|där per 1000 words (A1) | T↓ | T↑ (0.508) | +0.211 | ✗ LR |
| 6 | `mean_word_len` | mean word length (chars) (E5) | ? | T↑ (0.514) | +0.206 |  |
| 7 | `rate_pr_jag` | jag/mig/min per 1000 words (A11) | ? | T↑ (0.529) | +0.189 |  |
| 8 | `rate_inte_minst` | inte minst per 1000 words (A22) | T↓ | T↓ (0.496) | -0.180 |  |
| 9 | `r_kommer_att` | share kommer att among kommer att + kommer INF (A17) | T↑ | T↑ (0.523) | +0.172 |  |
| 10 | `rate_exclam` | exclamation marks (incl. vocatives like Herr talman!) per 1000 words (E9) | T↓ | T↓ (0.493) | -0.161 |  |
| 11 | `rate_pr_hanhon` | han/hon/honom/henne per 1000 words (A11) | ? | T↑ (0.513) | +0.151 |  |
| 12 | `rate_n_long15` | words >= 15 characters per 1000 words (C6) | ? | T↓ (0.497) | -0.146 |  |
| 13 | `rate_pr_ni` | ni/er/du per 1000 words (A11) | ? | T↑ (0.512) | +0.140 |  |
| 14 | `rate_conn_T` | T-leaning connectives (4.3a minus A3 core) per 1000 words (D1) | T↑ | T↓ (0.486) | -0.139 | ✗ data ✗ LR |
| 15 | `r_vilja` | share 'skulle vilja' among skulle vilja + vill (A13) | T↑ | T↑ (0.511) | +0.126 |  |
| 16 | `rate_av` | preposition av per 1000 words (A6) | T↑ | T↑ (0.547) | +0.125 |  |
| 17 | `rate_cap_month_nat` | capitalised nationality/month mid-sentence per 1000 words (F4) | T↑ | T↑ (0.515) | +0.125 |  |
| 18 | `log_n_words` | log number of words (-) | ? | T↑ (0.580) | -0.122 |  |
| 19 | `rate_pr_vi` | vi/oss/vår per 1000 words (A11) | ? | T↑ (0.508) | +0.121 |  |
| 20 | `share_si_men_och` | share of sentences: sentence-initial men/och/för (B9) | T↓ | T↑ (0.508) | +0.119 | ✗ LR |
| 21 | `rate_pr_de` | de/dem/deras per 1000 words (A11) | ? | T↑ (0.537) | +0.117 |  |
| 22 | `rate_colloq_core` | väldigt/lite per 1000 words (A4) | T↓ | T↓ (0.494) | -0.116 |  |
| 23 | `rate_tycka` | tycker/tyckte/tycka per 1000 words (A8) | T↓ | T↓ (0.490) | -0.116 |  |
| 24 | `share_si_adv_comma` | share of sentences: fronted adverbial + comma (Dessutom, ...) (B5) | T↑ | T↑ (0.502) | +0.109 |  |
| 25 | `r_att_kept` | share att kept after cognition verbs (C9) | T↑ | T↓ (0.496) | -0.109 | ✗ LR |

## gran = 100 (C = 0.03)

| # | feature | plain meaning | lit | data (AUC) | LR std. coef | |
|---:|---|---|---|---|---:|---|
| 1 | `mean_sent_len` | mean sentence length (words) (E1) | T↑ | T↑ (0.681) | +0.663 |  |
| 2 | `mean_word_len` | mean word length (chars) (E5) | ? | T↑ (0.539) | +0.571 |  |
| 3 | `rate_skall` | skall per 1000 words (A2) | T↑ | T↓ (0.375) | -0.498 | ✗ data ✗ LR |
| 4 | `rate_p_ju` | modal particle ju per 1000 words (A5) | T↓ | T↓ (0.455) | -0.475 |  |
| 5 | `rate_pr_jag` | jag/mig/min per 1000 words (A11) | ? | T↑ (0.546) | +0.431 |  |
| 6 | `rate_p_namligen` | nämligen per 1000 words (A5) | T↓ | T↓ (0.459) | -0.417 |  |
| 7 | `rate_inte_minst` | inte minst per 1000 words (A22) | T↓ | T↓ (0.474) | -0.385 |  |
| 8 | `rate_dem_O` | den/det/de här\|där per 1000 words (A1) | T↓ | T↑ (0.530) | +0.380 | ✗ data ✗ LR |
| 9 | `rate_dem_T` | denna/detta/dessa/denne per 1000 words (A1) | T↑ | T↑ (0.535) | +0.379 |  |
| 10 | `rate_n_long15` | words >= 15 characters per 1000 words (C6) | ? | T↓ (0.464) | -0.376 |  |
| 11 | `rate_conn_T` | T-leaning connectives (4.3a minus A3 core) per 1000 words (D1) | T↑ | T↓ (0.417) | -0.337 | ✗ data ✗ LR |
| 12 | `rate_naturligtvis` | naturligtvis/givetvis/självklart per 1000 words (A7) | T↑ | T↓ (0.471) | -0.336 | ✗ data ✗ LR |
| 13 | `rate_av` | preposition av per 1000 words (A6) | T↑ | T↑ (0.609) | +0.328 |  |
| 14 | `rate_pr_vi` | vi/oss/vår per 1000 words (A11) | ? | T↓ (0.490) | +0.314 |  |
| 15 | `r_anse_tycka` | share anse among anse+tycka (A8) | T↑ | T↑ (0.547) | +0.299 |  |
| 16 | `rate_pr_ni` | ni/er/du per 1000 words (A11) | ? | T↑ (0.537) | +0.292 |  |
| 17 | `rate_exclam` | exclamation marks (incl. vocatives like Herr talman!) per 1000 words (E9) | T↓ | T↓ (0.459) | -0.279 |  |
| 18 | `rate_pr_hanhon` | han/hon/honom/henne per 1000 words (A11) | ? | T↑ (0.544) | +0.279 |  |
| 19 | `share_si_men_och` | share of sentences: sentence-initial men/och/för (B9) | T↓ | T↑ (0.533) | +0.267 | ✗ data ✗ LR |
| 20 | `rate_poss_nonrefl` | hans/hennes/deras per 1000 words (A20) | T↑ | T↑ (0.550) | +0.266 |  |
| 21 | `rate_formal_misc` | samt/respektive/i enlighet med/i syfte att per 1000 words (A25) | T↑ | T↓ (0.478) | -0.262 | ✗ data ✗ LR |
| 22 | `rate_cap_month_nat` | capitalised nationality/month mid-sentence per 1000 words (F4) | T↑ | T↑ (0.555) | +0.257 |  |
| 23 | `rate_pr_de` | de/dem/deras per 1000 words (A11) | ? | T↑ (0.577) | +0.250 |  |
| 24 | `rate_colloq_core` | väldigt/lite per 1000 words (A4) | T↓ | T↓ (0.470) | -0.244 |  |
| 25 | `share_si_adv_comma` | share of sentences: fronted adverbial + comma (Dessutom, ...) (B5) | T↑ | T↑ (0.512) | +0.237 |  |

## gran = 500 (C = 0.1)

| # | feature | plain meaning | lit | data (AUC) | LR std. coef | |
|---:|---|---|---|---|---:|---|
| 1 | `mean_word_len` | mean word length (chars) (E5) | ? | T↑ (0.582) | +0.983 |  |
| 2 | `rate_p_ju` | modal particle ju per 1000 words (A5) | T↓ | T↓ (0.357) | -0.907 |  |
| 3 | `mean_sent_len` | mean sentence length (words) (E1) | T↑ | T↑ (0.788) | +0.879 |  |
| 4 | `rate_pr_jag` | jag/mig/min per 1000 words (A11) | ? | T↑ (0.568) | +0.703 |  |
| 5 | `rate_inte_minst` | inte minst per 1000 words (A22) | T↓ | T↓ (0.398) | -0.652 |  |
| 6 | `rate_p_namligen` | nämligen per 1000 words (A5) | T↓ | T↓ (0.351) | -0.639 |  |
| 7 | `rate_skall` | skall per 1000 words (A2) | T↑ | T↓ (0.245) | -0.598 | ✗ data ✗ LR |
| 8 | `rate_naturligtvis` | naturligtvis/givetvis/självklart per 1000 words (A7) | T↑ | T↓ (0.397) | -0.581 | ✗ data ✗ LR |
| 9 | `share_si_adv_comma` | share of sentences: fronted adverbial + comma (Dessutom, ...) (B5) | T↑ | T↑ (0.549) | +0.576 |  |
| 10 | `rate_conn_T` | T-leaning connectives (4.3a minus A3 core) per 1000 words (D1) | T↑ | T↓ (0.352) | -0.557 | ✗ data ✗ LR |
| 11 | `rate_dem_T` | denna/detta/dessa/denne per 1000 words (A1) | T↑ | T↑ (0.566) | +0.548 |  |
| 12 | `rate_formal_misc` | samt/respektive/i enlighet med/i syfte att per 1000 words (A25) | T↑ | T↓ (0.435) | -0.525 | ✗ data ✗ LR |
| 13 | `rate_pr_vi` | vi/oss/vår per 1000 words (A11) | ? | T↓ (0.483) | +0.524 |  |
| 14 | `rate_av` | preposition av per 1000 words (A6) | T↑ | T↑ (0.702) | +0.484 |  |
| 15 | `commas_per_sent` | commas per sentence (E2) | T↑ | T↑ (0.604) | +0.469 |  |
| 16 | `rate_n_long15` | words >= 15 characters per 1000 words (C6) | ? | T↓ (0.459) | -0.434 |  |
| 17 | `rate_i_sjalva_verket` | i själva verket per 1000 words (A22) | T↑ | T↑ (0.530) | +0.434 |  |
| 18 | `rate_pr_hanhon` | han/hon/honom/henne per 1000 words (A11) | ? | T↑ (0.619) | +0.423 |  |
| 19 | `rate_exclam` | exclamation marks (incl. vocatives like Herr talman!) per 1000 words (E9) | T↓ | T↓ (0.378) | -0.421 |  |
| 20 | `rate_cap_month_nat` | capitalised nationality/month mid-sentence per 1000 words (F4) | T↑ | T↑ (0.642) | +0.416 |  |
| 21 | `r_vilja` | share 'skulle vilja' among skulle vilja + vill (A13) | T↑ | T↑ (0.655) | +0.409 |  |
| 22 | `share_si_men_och` | share of sentences: sentence-initial men/och/för (B9) | T↓ | T↑ (0.603) | +0.408 | ✗ data ✗ LR |
| 23 | `rate_pr_de` | de/dem/deras per 1000 words (A11) | ? | T↑ (0.650) | +0.405 |  |
| 24 | `rate_egentligen` | egentligen per 1000 words (A22) | T↓ | T↓ (0.444) | -0.398 |  |
| 25 | `r_dem` | share denna-type among demonstratives (A1) | T↑ | T↑ (0.502) | -0.393 | ✗ LR |
