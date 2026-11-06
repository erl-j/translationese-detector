# Decision-tree rules over the expert features (candidate rules for the expert system)

Shallow trees (`tree_d3`, `tree_d4`) fitted on trn over the literature features (hybrid/features.py), un-scaled, so thresholds are in natural units: `rate_x` = hits per 1000 words, `share_x` = share of sentences, `r_x` = share of the T-leaning variant among the choices (smoothed: (k_T+.5)/(k_T+k_O+1)). Leaves are listed by how far they move the prior, weighted by size. P(trans) is the trn leaf purity (prior = .5).


## tree_d4, gran = sent

- IF mean_sent_len <= 26.2 AND rate_skall <= 39.2 AND rate_kommer_att <= 9.43 AND rate_pr_hanhon <= 31.9 THEN P(trans) = 0.46 (n = 59821)
- IF mean_sent_len > 26.2 AND rate_skall <= 23.5 AND rate_av > 10.1 AND mean_sent_len > 35.8 THEN P(trans) = 0.76 (n = 3957)
- IF mean_sent_len > 26.2 AND rate_skall <= 23.5 AND rate_av > 10.1 AND mean_sent_len <= 35.8 THEN P(trans) = 0.66 (n = 4444)
- IF mean_sent_len > 26.2 AND rate_skall <= 23.5 AND rate_av <= 10.1 AND rate_kommer_att <= 5.88 THEN P(trans) = 0.58 (n = 8681)
- IF mean_sent_len <= 26.2 AND rate_skall > 39.2 AND rate_pr_jag <= 31.7 AND rate_skall > 57.2 THEN P(trans) = 0.22 (n = 1872)
- IF mean_sent_len <= 26.2 AND rate_skall > 39.2 AND rate_pr_jag <= 31.7 AND rate_skall <= 57.2 THEN P(trans) = 0.30 (n = 1466)
- IF mean_sent_len <= 26.2 AND rate_skall <= 39.2 AND rate_kommer_att <= 9.43 AND rate_pr_hanhon > 31.9 THEN P(trans) = 0.72 (n = 1331)
- IF mean_sent_len > 26.2 AND rate_skall <= 23.5 AND rate_av <= 10.1 AND rate_kommer_att > 5.88 THEN P(trans) = 0.78 (n = 980)
- IF mean_sent_len <= 26.2 AND rate_skall <= 39.2 AND rate_kommer_att > 9.43 AND mattr50 > 0.911 THEN P(trans) = 0.64 (n = 1447)
- IF mean_sent_len <= 26.2 AND rate_skall <= 39.2 AND rate_kommer_att > 9.43 AND mattr50 <= 0.911 THEN P(trans) = 0.71 (n = 922)
- IF mean_sent_len > 26.2 AND rate_skall > 23.5 AND mean_word_len <= 5.28 THEN P(trans) = 0.34 (n = 1057)
- IF mean_sent_len <= 26.2 AND rate_skall > 39.2 AND rate_pr_jag > 31.7 THEN P(trans) = 0.34 (n = 888)
- IF mean_sent_len > 26.2 AND rate_skall > 23.5 AND mean_word_len > 5.28 THEN P(trans) = 0.46 (n = 1038)

## tree_d4, gran = 100

- IF mean_sent_len > 20.8 AND rate_skall <= 7.02 AND rate_p_namligen <= 6.83 AND rate_tycka <= 7.72 THEN P(trans) = 0.77 (n = 3929)
- IF mean_sent_len <= 20.8 AND rate_skall > 7.44 AND r_kommer_att <= 0.625 AND rate_av <= 22 THEN P(trans) = 0.18 (n = 1924)
- IF mean_sent_len <= 20.8 AND rate_skall <= 7.44 AND rate_p_ju <= 3.38 AND mean_sent_len <= 16.3 THEN P(trans) = 0.37 (n = 1782)
- IF mean_sent_len <= 20.8 AND rate_skall <= 7.44 AND rate_p_ju <= 3.38 AND mean_sent_len > 16.3 THEN P(trans) = 0.55 (n = 3028)
- IF mean_sent_len <= 20.8 AND rate_skall <= 7.44 AND rate_p_ju > 3.38 THEN P(trans) = 0.05 (n = 257)
- IF mean_sent_len > 20.8 AND rate_skall > 7.02 AND rate_av > 9.48 AND rate_skall <= 16.5 THEN P(trans) = 0.62 (n = 897)
- IF mean_sent_len > 20.8 AND rate_skall > 7.02 AND rate_av <= 9.48 AND rate_skall > 13.4 THEN P(trans) = 0.16 (n = 273)
- IF mean_sent_len <= 20.8 AND rate_skall > 7.44 AND r_kommer_att <= 0.625 AND rate_av > 22 THEN P(trans) = 0.34 (n = 459)
- IF mean_sent_len <= 20.8 AND rate_skall > 7.44 AND r_kommer_att > 0.625 AND rate_ande_part <= 9.57 THEN P(trans) = 0.33 (n = 361)
- IF mean_sent_len > 20.8 AND rate_skall > 7.02 AND rate_av <= 9.48 AND rate_skall <= 13.4 THEN P(trans) = 0.42 (n = 529)
- IF mean_sent_len > 20.8 AND rate_skall <= 7.02 AND rate_p_namligen > 6.83 THEN P(trans) = 0.30 (n = 175)
- IF mean_sent_len > 20.8 AND rate_skall > 7.02 AND rate_av > 9.48 AND rate_skall > 16.5 THEN P(trans) = 0.38 (n = 235)
- IF mean_sent_len > 20.8 AND rate_skall <= 7.02 AND rate_p_namligen <= 6.83 AND rate_tycka > 7.72 THEN P(trans) = 0.37 (n = 161)
- IF mean_sent_len <= 20.8 AND rate_skall > 7.44 AND r_kommer_att > 0.625 AND rate_ande_part > 9.57 THEN P(trans) = 0.56 (n = 160)

## tree_d4, gran = 500

- IF mean_sent_len > 19.7 AND rate_p_ju <= 1.86 AND rate_p_namligen <= 0.871 AND rate_skall <= 7.12 THEN P(trans) = 0.89 (n = 960)
- IF mean_sent_len <= 19.7 AND rate_skall > 3.78 AND rate_pr_hanhon <= 7.8 AND rate_av <= 24.2 THEN P(trans) = 0.10 (n = 756)
- IF mean_sent_len <= 19.7 AND rate_skall <= 3.78 AND rate_p_namligen <= 0.938 AND rate_inte_minst <= 1.93 THEN P(trans) = 0.64 (n = 307)
- IF mean_sent_len > 19.7 AND rate_p_ju > 1.86 AND rate_poss_nonrefl <= 1.86 AND rate_av <= 16.7 THEN P(trans) = 0.01 (n = 84)
- IF mean_sent_len > 19.7 AND rate_p_ju <= 1.86 AND rate_p_namligen > 0.871 AND rate_av <= 15.7 THEN P(trans) = 0.18 (n = 89)
- IF mean_sent_len <= 19.7 AND rate_skall <= 3.78 AND rate_p_namligen > 0.938 AND share_si_men_och <= 0.0328 THEN P(trans) = 0.04 (n = 57)
- IF mean_sent_len <= 19.7 AND rate_skall <= 3.78 AND rate_p_namligen <= 0.938 AND rate_inte_minst > 1.93 THEN P(trans) = 0.12 (n = 43)
- IF mean_sent_len > 19.7 AND rate_p_ju <= 1.86 AND rate_p_namligen > 0.871 AND rate_av > 15.7 THEN P(trans) = 0.63 (n = 76)
- IF mean_sent_len > 19.7 AND rate_p_ju > 1.86 AND rate_poss_nonrefl <= 1.86 AND rate_av > 16.7 THEN P(trans) = 0.22 (n = 27)
- IF mean_sent_len > 19.7 AND rate_p_ju > 1.86 AND rate_poss_nonrefl > 1.86 THEN P(trans) = 0.38 (n = 48)
- IF mean_sent_len <= 19.7 AND rate_skall <= 3.78 AND rate_p_namligen > 0.938 AND share_si_men_och > 0.0328 THEN P(trans) = 0.34 (n = 35)
- IF mean_sent_len <= 19.7 AND rate_skall > 3.78 AND rate_pr_hanhon > 7.8 THEN P(trans) = 0.66 (n = 32)
- IF mean_sent_len <= 19.7 AND rate_skall > 3.78 AND rate_pr_hanhon <= 7.8 AND rate_av > 24.2 THEN P(trans) = 0.44 (n = 70)
- IF mean_sent_len > 19.7 AND rate_p_ju <= 1.86 AND rate_p_namligen <= 0.871 AND rate_skall > 7.12 THEN P(trans) = 0.53 (n = 150)
