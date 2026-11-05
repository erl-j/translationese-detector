# What the expert-feature LR learned (`lr_expert`, fitted on dev-train)

Per granularity, the 25 features with the largest standardised coefficient (effect of +1 SD of the feature on the log-odds of *translated*). `lit` = predicted direction in docs/literature.md (T↑ more frequent in translations, T↓ less, ? = calibrate); `data` = direction of the class means on dev-train, with the feature's AUC alone (> .5 = higher in T); `LR` = sign of the coefficient given all other features. ✗ marks disagreement with the literature. LR signs of collinear features (e.g. a rate and its ratio) can differ from their univariate direction; trust `data` for the direction of a single cue, `LR` for its unique contribution.


## gran = sent (C = 0.01)

| # | feature | plain meaning | lit | data (AUC) | LR std. coef | |
|---:|---|---|---|---|---:|---|
| 1 | `mean_sent_len` | mean sentence length (words) (E1) | T↑ | T↑ (0.572) | +0.423 |  |
| 2 | `rate_p_ju` | modal particle ju per 1000 words (A5) | T↓ | T↓ (0.489) | -0.391 |  |
| 3 | `rate_p_namligen` | nämligen per 1000 words (A5) | T↓ | T↓ (0.492) | -0.283 |  |
| 4 | `r_skall` | share skall among skall+ska (A2) | T↑ | T↓ (0.472) | -0.232 | ✗ data ✗ LR |
| 5 | `rate_exclam` | exclamation marks (incl. vocatives like Herr talman!) per 1000 words (E9) | T↓ | T↓ (0.493) | -0.210 |  |
| 6 | `rate_inte_minst` | inte minst per 1000 words (A22) | T↓ | T↓ (0.496) | -0.195 |  |
| 7 | `rate_pr_jag` | jag/mig/min per 1000 words (A11) | ? | T↑ (0.532) | +0.191 |  |
| 8 | `rate_dem_O` | den/det/de här\|där per 1000 words (A1) | T↓ | T↑ (0.507) | +0.189 | ✗ LR |
| 9 | `mean_word_len` | mean word length (chars) (E5) | ? | T↑ (0.506) | +0.177 |  |
| 10 | `rate_pr_hanhon` | han/hon/honom/henne per 1000 words (A11) | ? | T↑ (0.514) | +0.171 |  |
| 11 | `rate_pr_ni` | ni/er/du per 1000 words (A11) | ? | T↑ (0.515) | +0.168 |  |
| 12 | `r_kommer_att` | share kommer att among kommer att + kommer INF (A17) | T↑ | T↑ (0.522) | +0.164 |  |
| 13 | `rate_pr_vi` | vi/oss/vår per 1000 words (A11) | ? | T↑ (0.519) | +0.155 |  |
| 14 | `rate_conn_T` | T-leaning connectives (4.3a minus A3 core) per 1000 words (D1) | T↑ | T↓ (0.483) | -0.149 | ✗ data ✗ LR |
| 15 | `log_n_words` | log number of words (-) | ? | T↑ (0.571) | -0.139 |  |
| 16 | `rate_colloq_core` | väldigt/lite per 1000 words (A4) | T↓ | T↓ (0.494) | -0.131 |  |
| 17 | `rate_n_long15` | words >= 15 characters per 1000 words (C6) | ? | T↓ (0.493) | -0.127 |  |
| 18 | `rate_cap_month_nat` | capitalised nationality/month mid-sentence per 1000 words (F4) | T↑ | T↑ (0.515) | +0.125 |  |
| 19 | `rate_av` | preposition av per 1000 words (A6) | T↑ | T↑ (0.544) | +0.124 |  |
| 20 | `share_si_men_och` | share of sentences: sentence-initial men/och/för (B9) | T↓ | T↑ (0.508) | +0.123 | ✗ LR |
| 21 | `share_si_adv_comma` | share of sentences: fronted adverbial + comma (Dessutom, ...) (B5) | T↑ | T↑ (0.503) | +0.123 |  |
| 22 | `r_vilja` | share 'skulle vilja' among skulle vilja + vill (A13) | T↑ | T↑ (0.512) | +0.119 |  |
| 23 | `r_anse_tycka` | share anse among anse+tycka (A8) | T↑ | T↑ (0.511) | +0.114 |  |
| 24 | `rate_pr_de` | de/dem/deras per 1000 words (A11) | ? | T↑ (0.534) | +0.111 |  |
| 25 | `rate_egentligen` | egentligen per 1000 words (A22) | T↓ | T↓ (0.498) | -0.109 |  |

## gran = 100 (C = 0.03)

| # | feature | plain meaning | lit | data (AUC) | LR std. coef | |
|---:|---|---|---|---|---:|---|
| 1 | `rate_p_ju` | modal particle ju per 1000 words (A5) | T↓ | T↓ (0.442) | -0.568 |  |
| 2 | `mean_sent_len` | mean sentence length (words) (E1) | T↑ | T↑ (0.660) | +0.521 |  |
| 3 | `rate_skall` | skall per 1000 words (A2) | T↑ | T↓ (0.375) | -0.501 | ✗ data ✗ LR |
| 4 | `mean_word_len` | mean word length (chars) (E5) | ? | T↑ (0.525) | +0.497 |  |
| 5 | `rate_p_namligen` | nämligen per 1000 words (A5) | T↓ | T↓ (0.455) | -0.457 |  |
| 6 | `rate_pr_jag` | jag/mig/min per 1000 words (A11) | ? | T↑ (0.561) | +0.422 |  |
| 7 | `rate_inte_minst` | inte minst per 1000 words (A22) | T↓ | T↓ (0.475) | -0.370 |  |
| 8 | `rate_dem_T` | denna/detta/dessa/denne per 1000 words (A1) | T↑ | T↑ (0.537) | +0.369 |  |
| 9 | `rate_exclam` | exclamation marks (incl. vocatives like Herr talman!) per 1000 words (E9) | T↓ | T↓ (0.455) | -0.364 |  |
| 10 | `rate_naturligtvis` | naturligtvis/givetvis/självklart per 1000 words (A7) | T↑ | T↓ (0.472) | -0.354 | ✗ data ✗ LR |
| 11 | `rate_pr_vi` | vi/oss/vår per 1000 words (A11) | ? | T↑ (0.511) | +0.336 |  |
| 12 | `rate_conn_T` | T-leaning connectives (4.3a minus A3 core) per 1000 words (D1) | T↑ | T↓ (0.411) | -0.335 | ✗ data ✗ LR |
| 13 | `rate_av` | preposition av per 1000 words (A6) | T↑ | T↑ (0.606) | +0.329 |  |
| 14 | `rate_n_long15` | words >= 15 characters per 1000 words (C6) | ? | T↓ (0.461) | -0.324 |  |
| 15 | `rate_pr_ni` | ni/er/du per 1000 words (A11) | ? | T↑ (0.544) | +0.321 |  |
| 16 | `rate_pr_hanhon` | han/hon/honom/henne per 1000 words (A11) | ? | T↑ (0.551) | +0.293 |  |
| 17 | `rate_dem_O` | den/det/de här\|där per 1000 words (A1) | T↓ | T↑ (0.527) | +0.288 | ✗ data ✗ LR |
| 18 | `rate_poss_nonrefl` | hans/hennes/deras per 1000 words (A20) | T↑ | T↑ (0.552) | +0.275 |  |
| 19 | `share_si_adv_comma` | share of sentences: fronted adverbial + comma (Dessutom, ...) (B5) | T↑ | T↑ (0.514) | +0.269 |  |
| 20 | `r_anse_tycka` | share anse among anse+tycka (A8) | T↑ | T↑ (0.550) | +0.265 |  |
| 21 | `rate_formal_misc` | samt/respektive/i enlighet med/i syfte att per 1000 words (A25) | T↑ | T↓ (0.472) | -0.262 | ✗ data ✗ LR |
| 22 | `share_si_men_och` | share of sentences: sentence-initial men/och/för (B9) | T↓ | T↑ (0.532) | +0.261 | ✗ data ✗ LR |
| 23 | `rate_cap_month_nat` | capitalised nationality/month mid-sentence per 1000 words (F4) | T↑ | T↑ (0.555) | +0.241 |  |
| 24 | `r_natur_ju` | share naturligtvis among naturligtvis+ju (A7) | T↑ | T↑ (0.527) | +0.236 |  |
| 25 | `rate_colloq_core` | väldigt/lite per 1000 words (A4) | T↓ | T↓ (0.469) | -0.233 |  |

## gran = 500 (C = 0.1)

| # | feature | plain meaning | lit | data (AUC) | LR std. coef | |
|---:|---|---|---|---|---:|---|
| 1 | `rate_p_ju` | modal particle ju per 1000 words (A5) | T↓ | T↓ (0.321) | -0.862 |  |
| 2 | `mean_word_len` | mean word length (chars) (E5) | ? | T↑ (0.559) | +0.840 |  |
| 3 | `mean_sent_len` | mean sentence length (words) (E1) | T↑ | T↑ (0.765) | +0.674 |  |
| 4 | `rate_inte_minst` | inte minst per 1000 words (A22) | T↓ | T↓ (0.404) | -0.666 |  |
| 5 | `rate_p_namligen` | nämligen per 1000 words (A5) | T↓ | T↓ (0.340) | -0.636 |  |
| 6 | `rate_pr_jag` | jag/mig/min per 1000 words (A11) | ? | T↑ (0.589) | +0.591 |  |
| 7 | `share_si_adv_comma` | share of sentences: fronted adverbial + comma (Dessutom, ...) (B5) | T↑ | T↑ (0.553) | +0.553 |  |
| 8 | `rate_dem_T` | denna/detta/dessa/denne per 1000 words (A1) | T↑ | T↑ (0.567) | +0.536 |  |
| 9 | `rate_pr_vi` | vi/oss/vår per 1000 words (A11) | ? | T↑ (0.524) | +0.532 |  |
| 10 | `rate_skall` | skall per 1000 words (A2) | T↑ | T↓ (0.242) | -0.527 | ✗ data ✗ LR |
| 11 | `rate_av` | preposition av per 1000 words (A6) | T↑ | T↑ (0.695) | +0.525 |  |
| 12 | `rate_conn_T` | T-leaning connectives (4.3a minus A3 core) per 1000 words (D1) | T↑ | T↓ (0.339) | -0.522 | ✗ data ✗ LR |
| 13 | `rate_formal_misc` | samt/respektive/i enlighet med/i syfte att per 1000 words (A25) | T↑ | T↓ (0.419) | -0.501 | ✗ data ✗ LR |
| 14 | `rate_naturligtvis` | naturligtvis/givetvis/självklart per 1000 words (A7) | T↑ | T↓ (0.415) | -0.497 | ✗ data ✗ LR |
| 15 | `rate_pr_hanhon` | han/hon/honom/henne per 1000 words (A11) | ? | T↑ (0.634) | +0.487 |  |
| 16 | `commas_per_sent` | commas per sentence (E2) | T↑ | T↑ (0.605) | +0.473 |  |
| 17 | `rate_i_sjalva_verket` | i själva verket per 1000 words (A22) | T↑ | T↑ (0.535) | +0.472 |  |
| 18 | `r_dem` | share denna-type among demonstratives (A1) | T↑ | T↑ (0.521) | -0.435 | ✗ LR |
| 19 | `share_si_men_och` | share of sentences: sentence-initial men/och/för (B9) | T↓ | T↑ (0.599) | +0.408 | ✗ data ✗ LR |
| 20 | `rate_cap_month_nat` | capitalised nationality/month mid-sentence per 1000 words (F4) | T↑ | T↑ (0.643) | +0.407 |  |
| 21 | `rate_exclam` | exclamation marks (incl. vocatives like Herr talman!) per 1000 words (E9) | T↓ | T↓ (0.387) | -0.403 |  |
| 22 | `share_si_subj` | share of sentences: subject-first: PRON (SADV) FIN (B1) | T↑ | T↑ (0.620) | +0.385 |  |
| 23 | `rate_egentligen` | egentligen per 1000 words (A22) | T↓ | T↓ (0.447) | -0.380 |  |
| 24 | `rate_poss_nonrefl` | hans/hennes/deras per 1000 words (A20) | T↑ | T↑ (0.684) | +0.373 |  |
| 25 | `rate_pr_ni` | ni/er/du per 1000 words (A11) | ? | T↑ (0.590) | +0.370 |  |
