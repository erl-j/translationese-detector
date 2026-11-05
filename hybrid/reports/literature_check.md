# Literature directions vs dev-train data

For every expert feature with a predicted direction in docs/literature.md: the feature's AUC alone on dev-train at 100- and 500-token granularity (> .5 = higher in translations), the class means at 500 tokens and the T/O ratio of the class means (rare cues have AUC ≈ .5 because most units have zero hits, so the ratio matters for them). A granularity *agrees with T↑* if AUC ≥ .53 or T/O ≥ 1.25 (and conversely AUC ≤ .47 or T/O ≤ 0.8 for T↓); the ratio only counts for items with ≥ 30 hits in dev-train at that granularity (`hits` column, 500). **Contradicted** = the data points the other way at both 100 and 500; **confirmed** = the predicted way at both; otherwise weak/mixed. Ratios (`r_*`) are the share of the T-leaning variant.

| feature | meaning | lit id | lit | AUC 100 | AUC 500 | mean O (500) | mean T (500) | T/O (500) | hits | verdict |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `r_skall` | share skall among skall+ska | A2 | T↑ | 0.376 | 0.242 | 0.836 | 0.705 | 0.84 | - | **contradicted** |
| `rate_skall` | skall per 1000 words | A2 | T↑ | 0.375 | 0.242 | 6.5 | 2.81 | 0.43 | 4437 | **contradicted** |
| `rate_conn_T` | T-leaning connectives (4.3a minus A3 core) per 1000 words | D1 | T↑ | 0.411 | 0.339 | 6.97 | 4.74 | 0.68 | 5490 | **contradicted** |
| `share_si_men_och` | share of sentences: sentence-initial men/och/för | B9 | T↓ | 0.532 | 0.599 | 0.0282 | 0.0451 | 1.60 | 1652 | **contradicted** |
| `rate_naturligtvis` | naturligtvis/givetvis/självklart per 1000 words | A7 | T↑ | 0.472 | 0.415 | 1.24 | 0.649 | 0.53 | 893 | **contradicted** |
| `rate_formal_misc` | samt/respektive/i enlighet med/i syfte att per 1000 words | A25 | T↑ | 0.472 | 0.419 | 1.19 | 0.664 | 0.56 | 878 | **contradicted** |
| `rate_att_kept` | cognition verb + att per 1000 words | C9 | T↑ | 0.467 | 0.419 | 3.07 | 2.26 | 0.74 | 2494 | **contradicted** |
| `rate_dem_O` | den/det/de här\|där per 1000 words | A1 | T↓ | 0.527 | 0.567 | 0.191 | 0.827 | 4.32 | 452 | **contradicted** |
| `share_si_conn_T` | share of sentences: sentence-initial T connective | D1 | T↑ | 0.465 | 0.448 | 0.0429 | 0.0354 | 0.82 | 1847 | **contradicted** |
| `rate_bli_pass` | bli-passive per 1000 words | C1 | T↑ | 0.492 | 0.456 | 1.01 | 0.714 | 0.71 | 808 | **contradicted** |
| `rate_tro_anse_1p` | jag/vi tror\|anser att per 1000 words | A8 | T↑ | 0.493 | 0.469 | 0.787 | 0.623 | 0.79 | 660 | **contradicted** |
| `rate_det_ar_darfor` | det är därför per 1000 words | B8 | T↑ | 0.498 | 0.487 | 0.2 | 0.141 | 0.71 | 160 | **contradicted** |
| `share_si_resump_sa` | share of sentences: resumptive så after fronted clause | B4 | T↓ | 0.503 | 0.505 | 0.00144 | 0.00227 | 1.58 | 81 | **contradicted** |
| `mean_sent_len` | mean sentence length (words) | E1 | T↑ | 0.660 | 0.765 | 18.8 | 22.3 | 1.18 | - | confirmed |
| `rate_av` | preposition av per 1000 words | A6 | T↑ | 0.606 | 0.695 | 13.8 | 19 | 1.38 | 15004 | confirmed |
| `rate_poss_nonrefl` | hans/hennes/deras per 1000 words | A20 | T↑ | 0.552 | 0.684 | 0.635 | 1.98 | 3.11 | 1168 | confirmed |
| `rate_kommer_att` | kommer att per 1000 words | A17 | T↑ | 0.572 | 0.682 | 1.72 | 3.73 | 2.17 | 2465 | confirmed |
| `rate_p_ju` | modal particle ju per 1000 words | A5 | T↓ | 0.442 | 0.321 | 1.23 | 0.0529 | 0.04 | 634 | confirmed |
| `r_kommer_att` | share kommer att among kommer att + kommer INF | A17 | T↑ | 0.573 | 0.679 | 0.651 | 0.75 | 1.15 | - | confirmed |
| `rate_man` | generic man per 1000 words | A9 | T↓ | 0.420 | 0.337 | 4.74 | 2.44 | 0.52 | 3403 | confirmed |
| `rate_p_namligen` | nämligen per 1000 words | A5 | T↓ | 0.455 | 0.340 | 0.99 | 0.143 | 0.14 | 553 | confirmed |
| `rate_tycka` | tycker/tyckte/tycka per 1000 words | A8 | T↓ | 0.451 | 0.346 | 1.35 | 0.26 | 0.19 | 782 | confirmed |
| `r_vilja` | share 'skulle vilja' among skulle vilja + vill | A13 | T↑ | 0.560 | 0.649 | 0.305 | 0.417 | 1.37 | - | confirmed |
| `r_subj_first` | share subject-first among subject-first+inverted | B1 | T↑ | 0.577 | 0.649 | 0.616 | 0.7 | 1.14 | - | confirmed |
| `rate_cap_month_nat` | capitalised nationality/month mid-sentence per 1000 words | F4 | T↑ | 0.555 | 0.643 | 1.03 | 2.47 | 2.39 | 1577 | confirmed |
| `r_anse_tycka` | share anse among anse+tycka | A8 | T↑ | 0.550 | 0.639 | 0.472 | 0.576 | 1.22 | - | confirmed |
| `rate_pr_de` | de/dem/deras per 1000 words | A11 | ? | 0.569 | 0.638 | 11.5 | 14.8 | 1.29 | 12110 | no prediction; data: T↑ |
| `rate_pr_hanhon` | han/hon/honom/henne per 1000 words | A11 | ? | 0.551 | 0.634 | 0.897 | 2.4 | 2.68 | 1481 | no prediction; data: T↑ |
| `share_si_subj` | share of sentences: subject-first: PRON (SADV) FIN | B1 | T↑ | 0.559 | 0.620 | 0.215 | 0.262 | 1.22 | 11061 | confirmed |
| `r_poss` | share hans/deras among possessives | A20 | T↑ | 0.530 | 0.616 | 0.37 | 0.46 | 1.24 | - | weak/mixed |
| `share_si_inverted` | share of sentences: topicalised: X ... FIN SUBJ (V2 inversion) | B1 | T↓ | 0.432 | 0.386 | 0.123 | 0.0953 | 0.77 | 5218 | confirmed |
| `rate_exclam` | exclamation marks (incl. vocatives like Herr talman!) per 1000 words | E9 | T↓ | 0.455 | 0.387 | 3.44 | 2.47 | 0.72 | 2767 | confirmed |
| `rate_skulle_vilja` | jag/vi skulle vilja per 1000 words | A13 | T↑ | 0.536 | 0.612 | 0.302 | 0.992 | 3.28 | 578 | confirmed |
| `commas_per_sent` | commas per sentence | E2 | T↑ | 0.564 | 0.605 | 0.631 | 0.779 | 1.23 | - | confirmed |
| `rate_colloq_core` | väldigt/lite per 1000 words | A4 | T↓ | 0.469 | 0.402 | 0.98 | 0.285 | 0.29 | 611 | confirmed |
| `rate_inte_minst` | inte minst per 1000 words | A22 | T↓ | 0.475 | 0.404 | 0.551 | 0.0738 | 0.13 | 306 | confirmed |
| `rate_pr_ni` | ni/er/du per 1000 words | A11 | ? | 0.544 | 0.590 | 1.23 | 2.63 | 2.15 | 1743 | no prediction; data: T↑ |
| `rate_pr_jag` | jag/mig/min per 1000 words | A11 | ? | 0.561 | 0.589 | 15 | 18.2 | 1.22 | 15280 | no prediction; data: T↑ |
| `rate_pied_rel` | pied-piped i/på/för vilket\|vilka; ', vilken/vilka' per 1000 words | A18 | T↑ | 0.528 | 0.587 | 0.614 | 1.18 | 1.92 | 813 | confirmed |
| `share_si_subclause` | share of sentences: starts with a subjunction | B1 | ? | 0.520 | 0.587 | 0.0359 | 0.0492 | 1.37 | 1912 | no prediction; data: T↑ |
| `rate_vill_1p` | jag/vi vill per 1000 words | A13 | T↓ | 0.466 | 0.416 | 2.87 | 2.11 | 0.74 | 2328 | confirmed |
| `r_natur_ju` | share naturligtvis among naturligtvis+ju | A7 | T↑ | 0.527 | 0.576 | 0.507 | 0.564 | 1.11 | - | weak/mixed |
| `rate_valkomnar` | välkomnar per 1000 words | G4 | T↑ | 0.521 | 0.576 | 0.243 | 0.694 | 2.86 | 419 | confirmed |
| `rate_p_alltsa` | alltså per 1000 words | A5 | T↓ | 0.480 | 0.425 | 0.533 | 0.135 | 0.25 | 322 | confirmed |
| `rate_calque` | calque lexicon hits (4.6) per 1000 words | A16 | T↑ | 0.525 | 0.572 | 0.68 | 1.16 | 1.71 | 838 | confirmed |
| `rate_p_anda` | ändå per 1000 words | A5 | T↓ | 0.482 | 0.428 | 0.586 | 0.176 | 0.30 | 367 | confirmed |
| `share_si_det_expl` | share of sentences: det är/finns/gäller... | B2 | ? | 0.458 | 0.432 | 0.0895 | 0.0741 | 0.83 | 3918 | no prediction; data: T↓ |
| `rate_dem_T` | denna/detta/dessa/denne per 1000 words | A1 | T↑ | 0.537 | 0.567 | 12.6 | 14.2 | 1.13 | 12399 | confirmed |
| `rate_ande_part` | -ande/-ende participles minus lexicalised stop-list per 1000 words | C2 | T↑ | 0.539 | 0.565 | 7.85 | 9.35 | 1.19 | 7921 | confirmed |
| `r_att_kept` | share att kept after cognition verbs | C9 | T↑ | 0.473 | 0.435 | 0.72 | 0.69 | 0.96 | - | weak/mixed |
| `rate_talman_addr` | Herr/Fru talman/ordförande address per 1000 words | G | ? | 0.478 | 0.436 | 3.11 | 2.71 | 0.87 | 2707 | no prediction; data: T↓ |
| `mean_word_len` | mean word length (chars) | E5 | ? | 0.525 | 0.559 | 5.38 | 5.44 | 1.01 | - | no prediction; data: T↑ |
| `rate_faktum_att` | det faktum att per 1000 words | A12 | T↑ | 0.515 | 0.557 | 0.0892 | 0.347 | 3.89 | 193 | confirmed |
| `rate_n_long15` | words >= 15 characters per 1000 words | C6 | ? | 0.461 | 0.444 | 28 | 26.1 | 0.93 | 25106 | no prediction; data: T↓ |
| `share_si_adv_comma` | share of sentences: fronted adverbial + comma (Dessutom, ...) | B5 | T↑ | 0.514 | 0.553 | 0.000741 | 0.00672 | 9.07 | 151 | confirmed |
| `rate_egentligen` | egentligen per 1000 words | A22 | T↓ | 0.486 | 0.447 | 0.337 | 0.0765 | 0.23 | 200 | confirmed |
| `rate_s_pass_modal` | modal + s-passive (måste göras) per 1000 words | C1 | ? | 0.483 | 0.448 | 3.05 | 2.45 | 0.80 | 2567 | no prediction; data: T↓ |
| `share_si_det_obj` | share of sentences: fronted object det (Det vet vi) | B3 | T↓ | 0.488 | 0.453 | 0.00458 | 0.000682 | 0.15 | 138 | confirmed |
| `rate_formal_ext` | extended archaic/formal list (icke, ty, sålunda, beträffande...) per 1000 words | A3 | T↑ | 0.514 | 0.546 | 0.526 | 0.857 | 1.63 | 629 | confirmed |
| `rate_s_pass` | -s verb forms (s-passive proxy) minus deponent stop-list per 1000 words | C1 | ? | 0.484 | 0.454 | 12.1 | 11 | 0.91 | 10732 | no prediction; data: T↓ |
| `rate_native_fixed` | native fixed expressions (4.6-4.7) per 1000 words | E10 | T↓ | 0.514 | 0.544 | 1.42 | 1.86 | 1.31 | 1508 | weak/mixed |
| `rate_particle_verb` | particle verbs (ta upp, lägga fram...) per 1000 words | 4.7 | T↓ | 0.515 | 0.542 | 2.3 | 2.75 | 1.19 | 2327 | weak/mixed |
| `rate_colloq_ext` | jätte-, ganska, kul, sån(t), nåt/nån... per 1000 words | A4 | T↓ | 0.491 | 0.459 | 0.523 | 0.3 | 0.57 | 388 | confirmed |
| `mattr50` | moving-average type/token ratio (50-word windows) | E3 | T↓ | 0.483 | 0.460 | 0.831 | 0.828 | 1.00 | - | weak/mixed |
| `rate_poss_refl` | sin/sitt/sina per 1000 words | A20 | T↓ | 0.515 | 0.539 | 2.35 | 2.66 | 1.13 | 2316 | weak/mixed |
| `rate_mycket_int` | mycket + adjective per 1000 words | A21 | T↑ | 0.514 | 0.537 | 0.641 | 0.846 | 1.32 | 683 | confirmed |
| `rate_i_sjalva_verket` | i själva verket per 1000 words | A22 | T↑ | 0.507 | 0.535 | 0.0461 | 0.211 | 4.58 | 114 | confirmed |
| `rate_sade_long` | sade per 1000 words | A23 | T↑ | 0.511 | 0.534 | 0.405 | 0.61 | 1.51 | 464 | confirmed |
| `rate_colon` | colons per 1000 words | E8 | ? | 0.488 | 0.468 | 1.69 | 1.38 | 0.82 | 1435 | no prediction; data: T↓ |
| `r_sade` | share sade among sade+sa | A23 | T↑ | 0.511 | 0.530 | 0.545 | 0.562 | 1.03 | - | weak/mixed |
| `rate_faktiskt` | faktiskt per 1000 words | A22 | ? | 0.493 | 0.470 | 0.616 | 0.461 | 0.75 | 505 | no prediction; data: T↓ |
| `rate_paren` | parentheses per 1000 words | D3 | T↑ | 0.509 | 0.527 | 0.523 | 0.87 | 1.66 | 620 | confirmed |
| `rate_helt_enkelt` | helt enkelt per 1000 words | A22 | ? | 0.506 | 0.526 | 0.0727 | 0.197 | 2.71 | 121 | no prediction; data: T↑ |
| `rate_conn_O` | liksom/i alla fall/över huvud taget per 1000 words | D2 | T↓ | 0.492 | 0.474 | 0.435 | 0.328 | 0.75 | 357 | confirmed |
| `rate_latinate_verb` | Latinate verbs (presentera, garantera...) per 1000 words | 4.7 | T↑ | 0.506 | 0.525 | 1.11 | 1.22 | 1.10 | 1073 | weak/mixed |
| `rate_p_val` | modal particle väl (after finite verb) per 1000 words | A5 | T↓ | 0.494 | 0.476 | 0.155 | 0.0483 | 0.31 | 98 | confirmed |
| `rate_pr_vi` | vi/oss/vår per 1000 words | A11 | ? | 0.511 | 0.524 | 18.8 | 19.8 | 1.06 | 17856 | no prediction; data: unclear |
| `rate_question` | question marks per 1000 words | E9 | T↓ | 0.495 | 0.477 | 1.73 | 1.52 | 0.88 | 1511 | weak/mixed |
| `r_dem` | share denna-type among demonstratives | A1 | T↑ | 0.527 | 0.521 | 0.905 | 0.873 | 0.96 | - | weak/mixed |
| `rate_att_dropped` | cognition verb + clause without att per 1000 words | C9 | T↓ | 0.496 | 0.480 | 0.137 | 0.0599 | 0.44 | 94 | confirmed |
| `rate_anse` | anser/ansåg/anse per 1000 words | A8 | T↑ | 0.506 | 0.518 | 0.974 | 1.05 | 1.08 | 933 | weak/mixed |
| `rate_en_adj_sak` | en ADJ sak per 1000 words | C4 | T↑ | 0.505 | 0.517 | 0.0645 | 0.13 | 2.01 | 88 | confirmed |
| `rate_haller_pa` | håller på att per 1000 words | C8 | T↑ | 0.503 | 0.516 | 0.0803 | 0.143 | 1.78 | 102 | confirmed |
| `r_latinate` | share Latinate among Latinate+particle verbs | 4.7 | T↑ | 0.490 | 0.485 | 0.405 | 0.396 | 0.98 | - | weak/mixed |
| `rate_formal_core` | emellertid/huruvida/således per 1000 words | A3 | T↑ | 0.496 | 0.487 | 0.815 | 0.75 | 0.92 | 727 | weak/mixed |
| `rate_lat_mig` | låt mig/oss per 1000 words | A14 | T↑ | 0.498 | 0.487 | 0.538 | 0.488 | 0.91 | 477 | weak/mixed |
| `log_n_words` | log number of words | - | ? | 0.530 | 0.488 | 6.23 | 6.23 | 1.00 | - | no prediction; data: unclear |
| `rate_semicolon` | semicolons per 1000 words | E8 | ? | 0.500 | 0.511 | 0.494 | 0.574 | 1.16 | 492 | no prediction; data: unclear |
| `rate_vara_pass` | vara + participle per 1000 words | C1 | T↑ | 0.501 | 0.511 | 0.847 | 0.935 | 1.10 | 823 | weak/mixed |
| `rate_verkligen` | verkligen per 1000 words | A22 | ? | 0.506 | 0.511 | 0.578 | 0.616 | 1.06 | 553 | no prediction; data: unclear |
| `rate_ska` | ska per 1000 words | A2 | T↓ | 0.502 | 0.510 | 0.00616 | 0.0628 | 10.20 | 30 | weak/mixed |
| `share_si_final_O` | share of sentences: sentence-final också/heller/ju/då/alltså | B10 | T↓ | 0.499 | 0.493 | 0.00157 | 0.00106 | 0.68 | 66 | confirmed |
| `rate_p_nog` | modal particle nog per 1000 words | A5 | T↓ | 0.500 | 0.493 | 0.162 | 0.132 | 0.82 | 137 | weak/mixed |
| `rate_comma` | commas (not decimal) per 1000 words | E2 | T↑ | 0.507 | 0.507 | 33.5 | 34.3 | 1.03 | 31310 | weak/mixed |
| `rate_vars` | vars per 1000 words | A19 | T↑ | 0.502 | 0.507 | 0.0868 | 0.118 | 1.36 | 94 | confirmed |
| `rate_def_gen` | definite s-genitive (kommissionens) per 1000 words | C5 | T↓ | 0.503 | 0.506 | 8.94 | 9.18 | 1.03 | 8385 | weak/mixed |
| `rate_kommer_kunna` | kommer att kunna/behöva per 1000 words | C7 | T↑ | 0.501 | 0.504 | 0.07 | 0.0882 | 1.26 | 73 | confirmed |
| `rate_sa_short` | sa (said) per 1000 words | A23 | T↓ | 0.501 | 0.504 | 0 | 0.0139 | 13912386.62 | 6 | weak/mixed |
| `rate_p_visst` | visst (not 'ett/i visst') per 1000 words | A5 | T↓ | 0.500 | 0.497 | 0.0402 | 0.0235 | 0.58 | 30 | confirmed |
| `rate_sedan_long` | sedan per 1000 words | A23 | T↑ | 0.504 | 0.497 | 0.735 | 0.728 | 0.99 | 678 | weak/mixed |
| `rate_role_pred` | är/som en ROLE per 1000 words | C3 | T↑ | 0.501 | 0.503 | 0.00413 | 0.0138 | 3.34 | 8 | weak/mixed |
| `rate_damer_herrar` | mina damer och herrar per 1000 words | A15 | ? | 0.500 | 0.502 | 0.0142 | 0.0253 | 1.78 | 18 | no prediction; data: unclear |
| `rate_kommer_inf` | kommer + infinitive (att dropped) per 1000 words | A17 | T↓ | 0.500 | 0.502 | 0.00401 | 0.0116 | 2.88 | 7 | weak/mixed |
| `rate_apos_gen` | apostrophe genitive per 1000 words | F1 | T↑ | 0.501 | 0.502 | 0.00192 | 0.0186 | 9.66 | 9 | weak/mixed |
| `rate_andes` | -andes participle per 1000 words | C2 | T↑ | 0.501 | 0.501 | 0.016 | 0.0252 | 1.57 | 19 | weak/mixed |
| `rate_short_other` | nåt/nån/sån/mej/dej/sej/dom per 1000 words | A23 | T↓ | 0.500 | 0.499 | 0.0223 | 0.0209 | 0.94 | 20 | weak/mixed |
| `rate_sen_short` | sen (since/then) per 1000 words | A23 | T↓ | 0.500 | 0.501 | 0.00407 | 0.00704 | 1.73 | 5 | weak/mixed |
| `rate_pct_nospace` | % without space per 1000 words | F7 | T↑ | 0.500 | 0.499 | 0.00206 | 0 | 0.00 | 1 | weak/mixed |
| `share_si_v3` | share of sentences: V3: SUBJ SADV FIN (Vi också vill) | B6 | T↑ | 0.500 | 0.499 | 3.66e-05 | 0 | 0.00 | 1 | weak/mixed |
| `rate_ens_gen` | generic en/ens per 1000 words | A10 | T↑ | 0.500 | 0.500 | 0 | 0 | 1.00 | 0 | weak/mixed |
