# Literature directions vs trn data

For every expert feature with a predicted direction in docs/literature.md: the feature's AUC alone on trn at 100- and 500-token granularity (> .5 = higher in translations), the class means at 500 tokens and the T/O ratio of the class means (rare cues have AUC ≈ .5 because most units have zero hits, so the ratio matters for them). A granularity *agrees with T↑* if AUC ≥ .53 or T/O ≥ 1.25 (and conversely AUC ≤ .47 or T/O ≤ 0.8 for T↓); the ratio only counts for items with ≥ 30 hits in trn at that granularity (`hits` column, 500). **Contradicted** = the data points the other way at both 100 and 500; **confirmed** = the predicted way at both; otherwise weak/mixed. Ratios (`r_*`) are the share of the T-leaning variant.

| feature | meaning | lit id | lit | AUC 100 | AUC 500 | mean O (500) | mean T (500) | T/O (500) | hits | verdict |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `rate_skall` | skall per 1000 words | A2 | T↑ | 0.375 | 0.245 | 6.43 | 2.79 | 0.43 | 6409 | **contradicted** |
| `r_skall` | share skall among skall+ska | A2 | T↑ | 0.377 | 0.245 | 0.834 | 0.708 | 0.85 | - | **contradicted** |
| `rate_conn_T` | T-leaning connectives (4.3a minus A3 core) per 1000 words | D1 | T↑ | 0.417 | 0.352 | 7.02 | 4.96 | 0.71 | 8310 | **contradicted** |
| `rate_naturligtvis` | naturligtvis/givetvis/självklart per 1000 words | A7 | T↑ | 0.471 | 0.397 | 1.2 | 0.529 | 0.44 | 1201 | **contradicted** |
| `share_si_men_och` | share of sentences: sentence-initial men/och/för | B9 | T↓ | 0.533 | 0.603 | 0.0263 | 0.0431 | 1.64 | 2380 | **contradicted** |
| `rate_att_kept` | cognition verb + att per 1000 words | C9 | T↑ | 0.461 | 0.400 | 3.1 | 2.16 | 0.70 | 3657 | **contradicted** |
| `rate_dem_O` | den/det/de här\|där per 1000 words | A1 | T↓ | 0.530 | 0.581 | 0.183 | 0.921 | 5.03 | 767 | **contradicted** |
| `r_att_kept` | share att kept after cognition verbs | C9 | T↑ | 0.468 | 0.421 | 0.722 | 0.683 | 0.95 | - | **contradicted** |
| `rate_formal_misc` | samt/respektive/i enlighet med/i syfte att per 1000 words | A25 | T↑ | 0.478 | 0.435 | 1.1 | 0.678 | 0.61 | 1236 | **contradicted** |
| `rate_bli_pass` | bli-passive per 1000 words | C1 | T↑ | 0.489 | 0.446 | 1.04 | 0.685 | 0.66 | 1197 | **contradicted** |
| `rate_tro_anse_1p` | jag/vi tror\|anser att per 1000 words | A8 | T↑ | 0.490 | 0.457 | 0.804 | 0.581 | 0.72 | 964 | **contradicted** |
| `rate_det_ar_darfor` | det är därför per 1000 words | B8 | T↑ | 0.497 | 0.488 | 0.199 | 0.143 | 0.72 | 237 | **contradicted** |
| `share_si_resump_sa` | share of sentences: resumptive så after fronted clause | B4 | T↓ | 0.502 | 0.504 | 0.00115 | 0.00183 | 1.59 | 99 | **contradicted** |
| `rate_ska` | ska per 1000 words | A2 | T↓ | 0.501 | 0.504 | 0.0146 | 0.039 | 2.67 | 37 | **contradicted** |
| `mean_sent_len` | mean sentence length (words) | E1 | T↑ | 0.681 | 0.788 | 18.6 | 22.2 | 1.19 | - | confirmed |
| `rate_av` | preposition av per 1000 words | A6 | T↑ | 0.609 | 0.702 | 13.6 | 19 | 1.39 | 22644 | confirmed |
| `rate_kommer_att` | kommer att per 1000 words | A17 | T↑ | 0.575 | 0.691 | 1.64 | 3.7 | 2.26 | 3707 | confirmed |
| `r_kommer_att` | share kommer att among kommer att + kommer INF | A17 | T↑ | 0.576 | 0.687 | 0.647 | 0.749 | 1.16 | - | confirmed |
| `rate_poss_nonrefl` | hans/hennes/deras per 1000 words | A20 | T↑ | 0.550 | 0.667 | 0.696 | 1.93 | 2.78 | 1827 | confirmed |
| `rate_tycka` | tycker/tyckte/tycka per 1000 words | A8 | T↓ | 0.450 | 0.337 | 1.33 | 0.224 | 0.17 | 1084 | confirmed |
| `r_vilja` | share 'skulle vilja' among skulle vilja + vill | A13 | T↑ | 0.558 | 0.655 | 0.302 | 0.421 | 1.39 | - | confirmed |
| `rate_pr_de` | de/dem/deras per 1000 words | A11 | ? | 0.577 | 0.650 | 11.5 | 15 | 1.31 | 18367 | no prediction; data: T↑ |
| `rate_p_namligen` | nämligen per 1000 words | A5 | T↓ | 0.459 | 0.351 | 0.912 | 0.143 | 0.16 | 733 | confirmed |
| `rate_p_ju` | modal particle ju per 1000 words | A5 | T↓ | 0.455 | 0.357 | 0.961 | 0.0544 | 0.06 | 707 | confirmed |
| `rate_cap_month_nat` | capitalised nationality/month mid-sentence per 1000 words | F4 | T↑ | 0.555 | 0.642 | 1.03 | 2.5 | 2.44 | 2447 | confirmed |
| `rate_man` | generic man per 1000 words | A9 | T↓ | 0.434 | 0.360 | 4.4 | 2.43 | 0.55 | 4749 | confirmed |
| `r_subj_first` | share subject-first among subject-first+inverted | B1 | T↑ | 0.561 | 0.634 | 0.625 | 0.699 | 1.12 | - | confirmed |
| `r_anse_tycka` | share anse among anse+tycka | A8 | T↑ | 0.547 | 0.629 | 0.483 | 0.581 | 1.20 | - | confirmed |
| `share_si_inverted` | share of sentences: topicalised: X ... FIN SUBJ (V2 inversion) | B1 | T↓ | 0.433 | 0.378 | 0.122 | 0.0933 | 0.76 | 7674 | confirmed |
| `rate_exclam` | exclamation marks (incl. vocatives like Herr talman!) per 1000 words | E9 | T↓ | 0.459 | 0.378 | 3.5 | 2.49 | 0.71 | 4156 | confirmed |
| `rate_pr_hanhon` | han/hon/honom/henne per 1000 words | A11 | ? | 0.544 | 0.619 | 0.923 | 2.21 | 2.40 | 2181 | no prediction; data: T↑ |
| `rate_skulle_vilja` | jag/vi skulle vilja per 1000 words | A13 | T↑ | 0.532 | 0.606 | 0.279 | 0.941 | 3.38 | 848 | confirmed |
| `commas_per_sent` | commas per sentence | E2 | T↑ | 0.568 | 0.604 | 0.612 | 0.748 | 1.22 | - | confirmed |
| `rate_inte_minst` | inte minst per 1000 words | A22 | T↓ | 0.474 | 0.398 | 0.583 | 0.0775 | 0.13 | 460 | confirmed |
| `rate_vill_1p` | jag/vi vill per 1000 words | A13 | T↓ | 0.466 | 0.402 | 2.87 | 2 | 0.70 | 3381 | confirmed |
| `rate_colloq_core` | väldigt/lite per 1000 words | A4 | T↓ | 0.470 | 0.405 | 0.944 | 0.287 | 0.30 | 857 | confirmed |
| `r_poss` | share hans/deras among possessives | A20 | T↑ | 0.524 | 0.595 | 0.377 | 0.452 | 1.20 | - | weak/mixed |
| `share_si_subclause` | share of sentences: starts with a subjunction | B1 | ? | 0.522 | 0.594 | 0.036 | 0.0505 | 1.41 | 2951 | no prediction; data: T↑ |
| `rate_talman_addr` | Herr/Fru talman/ordförande address per 1000 words | G | ? | 0.476 | 0.414 | 3.2 | 2.63 | 0.82 | 4048 | no prediction; data: T↓ |
| `share_si_det_expl` | share of sentences: det är/finns/gäller... | B2 | ? | 0.454 | 0.415 | 0.0888 | 0.0695 | 0.78 | 5673 | no prediction; data: T↓ |
| `mean_word_len` | mean word length (chars) | E5 | ? | 0.539 | 0.582 | 5.39 | 5.48 | 1.02 | - | no prediction; data: T↑ |
| `share_si_subj` | share of sentences: subject-first: PRON (SADV) FIN | B1 | T↑ | 0.539 | 0.580 | 0.223 | 0.255 | 1.14 | 16765 | confirmed |
| `rate_valkomnar` | välkomnar per 1000 words | G4 | T↑ | 0.521 | 0.578 | 0.235 | 0.692 | 2.94 | 644 | confirmed |
| `rate_pied_rel` | pied-piped i/på/för vilket\|vilka; ', vilken/vilka' per 1000 words | A18 | T↑ | 0.525 | 0.577 | 0.576 | 1.06 | 1.84 | 1135 | confirmed |
| `rate_pr_ni` | ni/er/du per 1000 words | A11 | ? | 0.537 | 0.573 | 1.26 | 2.35 | 1.86 | 2513 | no prediction; data: T↑ |
| `rate_p_anda` | ändå per 1000 words | A5 | T↓ | 0.482 | 0.428 | 0.569 | 0.185 | 0.33 | 525 | confirmed |
| `rate_pr_jag` | jag/mig/min per 1000 words | A11 | ? | 0.546 | 0.568 | 14.8 | 17.3 | 1.17 | 22326 | no prediction; data: T↑ |
| `rate_ande_part` | -ande/-ende participles minus lexicalised stop-list per 1000 words | C2 | T↑ | 0.536 | 0.568 | 7.58 | 9.02 | 1.19 | 11528 | confirmed |
| `rate_p_alltsa` | alltså per 1000 words | A5 | T↓ | 0.483 | 0.433 | 0.464 | 0.113 | 0.24 | 400 | confirmed |
| `rate_dem_T` | denna/detta/dessa/denne per 1000 words | A1 | T↑ | 0.535 | 0.566 | 12.5 | 14.1 | 1.13 | 18529 | confirmed |
| `rate_s_pass_modal` | modal + s-passive (måste göras) per 1000 words | C1 | ? | 0.479 | 0.437 | 3.14 | 2.47 | 0.78 | 3895 | no prediction; data: T↓ |
| `rate_calque` | calque lexicon hits (4.6) per 1000 words | A16 | T↑ | 0.519 | 0.559 | 0.637 | 0.997 | 1.56 | 1135 | confirmed |
| `rate_egentligen` | egentligen per 1000 words | A22 | T↓ | 0.486 | 0.444 | 0.343 | 0.0718 | 0.21 | 288 | confirmed |
| `rate_native_fixed` | native fixed expressions (4.6-4.7) per 1000 words | E10 | T↓ | 0.516 | 0.552 | 1.4 | 1.87 | 1.33 | 2271 | weak/mixed |
| `rate_faktum_att` | det faktum att per 1000 words | A12 | T↑ | 0.514 | 0.552 | 0.094 | 0.331 | 3.52 | 294 | confirmed |
| `share_si_adv_comma` | share of sentences: fronted adverbial + comma (Dessutom, ...) | B5 | T↑ | 0.512 | 0.549 | 0.000918 | 0.00678 | 7.38 | 250 | confirmed |
| `rate_particle_verb` | particle verbs (ta upp, lägga fram...) per 1000 words | 4.7 | T↓ | 0.519 | 0.548 | 2.24 | 2.74 | 1.22 | 3460 | weak/mixed |
| `rate_poss_refl` | sin/sitt/sina per 1000 words | A20 | T↓ | 0.519 | 0.548 | 2.33 | 2.72 | 1.17 | 3513 | weak/mixed |
| `share_si_det_obj` | share of sentences: fronted object det (Det vet vi) | B3 | T↓ | 0.489 | 0.455 | 0.00441 | 0.000818 | 0.19 | 197 | confirmed |
| `rate_faktiskt` | faktiskt per 1000 words | A22 | ? | 0.489 | 0.455 | 0.625 | 0.387 | 0.62 | 704 | no prediction; data: T↓ |
| `mattr50` | moving-average type/token ratio (50-word windows) | E3 | T↓ | 0.483 | 0.458 | 0.832 | 0.829 | 1.00 | - | weak/mixed |
| `rate_colloq_ext` | jätte-, ganska, kul, sån(t), nåt/nån... per 1000 words | A4 | T↓ | 0.490 | 0.458 | 0.468 | 0.246 | 0.53 | 496 | confirmed |
| `rate_n_long15` | words >= 15 characters per 1000 words | C6 | ? | 0.464 | 0.459 | 28.9 | 27.4 | 0.95 | 39020 | no prediction; data: T↓ |
| `rate_formal_ext` | extended archaic/formal list (icke, ty, sålunda, beträffande...) per 1000 words | A3 | T↑ | 0.513 | 0.540 | 0.525 | 0.821 | 1.56 | 933 | confirmed |
| `rate_sade_long` | sade per 1000 words | A23 | T↑ | 0.513 | 0.535 | 0.404 | 0.622 | 1.54 | 714 | confirmed |
| `r_sade` | share sade among sade+sa | A23 | T↑ | 0.513 | 0.534 | 0.544 | 0.563 | 1.03 | - | weak/mixed |
| `rate_s_pass` | -s verb forms (s-passive proxy) minus deponent stop-list per 1000 words | C1 | ? | 0.487 | 0.470 | 12.1 | 11.3 | 0.94 | 16226 | no prediction; data: T↓ |
| `rate_i_sjalva_verket` | i själva verket per 1000 words | A22 | T↑ | 0.506 | 0.530 | 0.0357 | 0.173 | 4.83 | 145 | confirmed |
| `rate_latinate_verb` | Latinate verbs (presentera, garantera...) per 1000 words | 4.7 | T↑ | 0.509 | 0.530 | 1.05 | 1.23 | 1.17 | 1584 | weak/mixed |
| `share_si_conn_T` | share of sentences: sentence-initial T connective | D1 | T↑ | 0.473 | 0.471 | 0.0411 | 0.0378 | 0.92 | 2769 | weak/mixed |
| `r_natur_ju` | share naturligtvis among naturligtvis+ju | A7 | T↑ | 0.513 | 0.528 | 0.527 | 0.552 | 1.05 | - | weak/mixed |
| `rate_att_dropped` | cognition verb + clause without att per 1000 words | C9 | T↓ | 0.494 | 0.472 | 0.162 | 0.0428 | 0.26 | 143 | confirmed |
| `rate_paren` | parentheses per 1000 words | D3 | T↑ | 0.507 | 0.527 | 0.498 | 0.737 | 1.48 | 842 | confirmed |
| `rate_conn_O` | liksom/i alla fall/över huvud taget per 1000 words | D2 | T↓ | 0.492 | 0.474 | 0.44 | 0.32 | 0.73 | 528 | confirmed |
| `rate_vara_pass` | vara + participle per 1000 words | C1 | T↑ | 0.504 | 0.525 | 0.817 | 0.965 | 1.18 | 1238 | weak/mixed |
| `rate_p_val` | modal particle väl (after finite verb) per 1000 words | A5 | T↓ | 0.495 | 0.477 | 0.137 | 0.0358 | 0.26 | 120 | confirmed |
| `rate_helt_enkelt` | helt enkelt per 1000 words | A22 | ? | 0.505 | 0.523 | 0.069 | 0.173 | 2.50 | 168 | no prediction; data: T↑ |
| `rate_mycket_int` | mycket + adjective per 1000 words | A21 | T↑ | 0.505 | 0.519 | 0.737 | 0.827 | 1.12 | 1087 | weak/mixed |
| `log_n_words` | log number of words | - | ? | 0.532 | 0.482 | 6.23 | 6.23 | 1.00 | - | no prediction; data: unclear |
| `rate_pr_vi` | vi/oss/vår per 1000 words | A11 | ? | 0.490 | 0.483 | 19.9 | 19.3 | 0.97 | 27214 | no prediction; data: unclear |
| `rate_def_gen` | definite s-genitive (kommissionens) per 1000 words | C5 | T↓ | 0.508 | 0.515 | 8.9 | 9.22 | 1.03 | 12576 | weak/mixed |
| `r_latinate` | share Latinate among Latinate+particle verbs | 4.7 | T↑ | 0.490 | 0.486 | 0.405 | 0.397 | 0.98 | - | weak/mixed |
| `rate_p_nog` | modal particle nog per 1000 words | A5 | T↓ | 0.499 | 0.487 | 0.177 | 0.124 | 0.70 | 209 | weak/mixed |
| `rate_haller_pa` | håller på att per 1000 words | C8 | T↑ | 0.503 | 0.513 | 0.0761 | 0.126 | 1.66 | 141 | confirmed |
| `rate_en_adj_sak` | en ADJ sak per 1000 words | C4 | T↑ | 0.503 | 0.512 | 0.0619 | 0.114 | 1.83 | 122 | confirmed |
| `rate_colon` | colons per 1000 words | E8 | ? | 0.495 | 0.488 | 1.49 | 1.31 | 0.88 | 1942 | no prediction; data: unclear |
| `rate_semicolon` | semicolons per 1000 words | E8 | ? | 0.502 | 0.511 | 0.398 | 0.482 | 1.21 | 611 | no prediction; data: unclear |
| `rate_vars` | vars per 1000 words | A19 | T↑ | 0.503 | 0.510 | 0.0821 | 0.121 | 1.47 | 141 | confirmed |
| `rate_question` | question marks per 1000 words | E9 | T↓ | 0.496 | 0.490 | 2.08 | 2 | 0.96 | 2839 | weak/mixed |
| `rate_p_visst` | visst (not 'ett/i visst') per 1000 words | A5 | T↓ | 0.498 | 0.491 | 0.0662 | 0.0175 | 0.26 | 58 | confirmed |
| `rate_lat_mig` | låt mig/oss per 1000 words | A14 | T↑ | 0.499 | 0.492 | 0.507 | 0.458 | 0.90 | 671 | weak/mixed |
| `share_si_final_O` | share of sentences: sentence-final också/heller/ju/då/alltså | B10 | T↓ | 0.498 | 0.492 | 0.00158 | 0.00106 | 0.67 | 99 | confirmed |
| `rate_comma` | commas (not decimal) per 1000 words | E2 | T↑ | 0.503 | 0.493 | 32.9 | 33.2 | 1.01 | 45826 | weak/mixed |
| `rate_kommer_kunna` | kommer att kunna/behöva per 1000 words | C7 | T↑ | 0.501 | 0.504 | 0.0759 | 0.0977 | 1.29 | 121 | confirmed |
| `rate_anse` | anser/ansåg/anse per 1000 words | A8 | T↑ | 0.501 | 0.504 | 1.13 | 1.08 | 0.96 | 1532 | weak/mixed |
| `r_dem` | share denna-type among demonstratives | A1 | T↑ | 0.522 | 0.502 | 0.905 | 0.869 | 0.96 | - | weak/mixed |
| `rate_apos_gen` | apostrophe genitive per 1000 words | F1 | T↑ | 0.501 | 0.502 | 0.00428 | 0.0254 | 5.94 | 21 | weak/mixed |
| `rate_kommer_inf` | kommer + infinitive (att dropped) per 1000 words | A17 | T↓ | 0.500 | 0.502 | 0.00286 | 0.01 | 3.50 | 9 | weak/mixed |
| `rate_damer_herrar` | mina damer och herrar per 1000 words | A15 | ? | 0.500 | 0.501 | 0.0101 | 0.0171 | 1.69 | 19 | no prediction; data: unclear |
| `rate_sa_short` | sa (said) per 1000 words | A23 | T↓ | 0.500 | 0.501 | 0.00435 | 0.0101 | 2.31 | 10 | weak/mixed |
| `rate_formal_core` | emellertid/huruvida/således per 1000 words | A3 | T↑ | 0.500 | 0.499 | 0.768 | 0.774 | 1.01 | 1070 | weak/mixed |
| `rate_andes` | -andes participle per 1000 words | C2 | T↑ | 0.500 | 0.501 | 0.0158 | 0.0228 | 1.45 | 27 | weak/mixed |
| `rate_verkligen` | verkligen per 1000 words | A22 | ? | 0.503 | 0.499 | 0.594 | 0.578 | 0.97 | 815 | no prediction; data: unclear |
| `rate_role_pred` | är/som en ROLE per 1000 words | C3 | T↑ | 0.500 | 0.501 | 0.00583 | 0.00855 | 1.47 | 10 | weak/mixed |
| `rate_sedan_long` | sedan per 1000 words | A23 | T↑ | 0.505 | 0.499 | 0.724 | 0.743 | 1.03 | 1019 | weak/mixed |
| `rate_sen_short` | sen (since/then) per 1000 words | A23 | T↓ | 0.500 | 0.500 | 0.0029 | 0.00437 | 1.51 | 5 | weak/mixed |
| `rate_pct_nospace` | % without space per 1000 words | F7 | T↑ | 0.500 | 0.500 | 0.00147 | 0 | 0.00 | 1 | weak/mixed |
| `share_si_v3` | share of sentences: V3: SUBJ SADV FIN (Vi också vill) | B6 | T↑ | 0.500 | 0.500 | 2.61e-05 | 0 | 0.00 | 1 | weak/mixed |
| `rate_short_other` | nåt/nån/sån/mej/dej/sej/dom per 1000 words | A23 | T↓ | 0.500 | 0.500 | 0.0202 | 0.0216 | 1.07 | 29 | weak/mixed |
| `rate_ens_gen` | generic en/ens per 1000 words | A10 | T↑ | 0.500 | 0.500 | 0 | 0 | 1.00 | 0 | weak/mixed |
