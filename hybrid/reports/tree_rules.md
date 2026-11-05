# Decision-tree rules over the expert features (candidate rules for the expert system)

Shallow trees (`tree_d3`, `tree_d4`) fitted on dev-train over the literature features (hybrid/features.py), un-scaled, so thresholds are in natural units: `rate_x` = hits per 1000 words, `share_x` = share of sentences, `r_x` = share of the T-leaning variant among the choices (smoothed: (k_T+.5)/(k_T+k_O+1)). Leaves are listed by how far they move the prior, weighted by size. P(trans) is the dev-train leaf purity (prior ≈ .47).


## tree_d4, gran = sent

- IF mean_sent_len > 26.8 AND rate_skall <= 21.5 AND rate_man <= 25.3 AND r_kommer_att <= 0.625 THEN P(trans) = 0.62 (n = 9860)
- IF mean_sent_len <= 26.8 AND rate_skall <= 39.2 AND rate_pr_hanhon <= 17.2 AND rate_pr_ni <= 24.9 THEN P(trans) = 0.44 (n = 40237)
- IF mean_sent_len <= 26.8 AND rate_skall > 39.2 AND rate_pr_jag <= 39.2 AND rate_skall > 51.3 THEN P(trans) = 0.21 (n = 1542)
- IF mean_sent_len > 26.8 AND rate_skall <= 21.5 AND rate_man <= 25.3 AND r_kommer_att > 0.625 THEN P(trans) = 0.79 (n = 1182)
- IF mean_sent_len <= 26.8 AND rate_skall <= 39.2 AND rate_pr_hanhon <= 17.2 AND rate_pr_ni > 24.9 THEN P(trans) = 0.68 (n = 1146)
- IF mean_sent_len <= 26.8 AND rate_skall <= 39.2 AND rate_pr_hanhon > 17.2 THEN P(trans) = 0.73 (n = 945)
- IF mean_sent_len > 26.8 AND rate_skall > 21.5 AND mean_word_len <= 5.34 THEN P(trans) = 0.30 (n = 873)
- IF mean_sent_len <= 26.8 AND rate_skall > 39.2 AND rate_pr_jag <= 39.2 AND rate_skall <= 51.3 THEN P(trans) = 0.28 (n = 726)
- IF mean_sent_len <= 26.8 AND rate_skall > 39.2 AND rate_pr_jag > 39.2 THEN P(trans) = 0.33 (n = 651)
- IF mean_sent_len > 26.8 AND rate_skall <= 21.5 AND rate_man > 25.3 THEN P(trans) = 0.42 (n = 895)
- IF mean_sent_len > 26.8 AND rate_skall > 21.5 AND mean_word_len > 5.34 THEN P(trans) = 0.44 (n = 664)

## tree_d4, gran = 100

- IF rate_skall <= 8.37 AND rate_p_ju <= 6.65 AND mean_sent_len > 21.2 AND rate_p_namligen <= 6.9 THEN P(trans) = 0.73 (n = 2601)
- IF rate_skall > 8.37 AND mean_sent_len <= 19.8 AND rate_pr_ni <= 8.58 AND rate_av <= 17.6 THEN P(trans) = 0.13 (n = 961)
- IF rate_skall <= 8.37 AND rate_p_ju <= 6.65 AND mean_sent_len <= 21.2 AND rate_p_namligen <= 3.52 THEN P(trans) = 0.50 (n = 3344)
- IF rate_skall > 8.37 AND mean_sent_len <= 19.8 AND rate_pr_ni <= 8.58 AND rate_av > 17.6 THEN P(trans) = 0.28 (n = 506)
- IF rate_skall <= 8.37 AND rate_p_ju > 6.65 AND rate_conn_T > 8.1 THEN P(trans) = 0.01 (n = 185)
- IF rate_skall > 8.37 AND mean_sent_len > 19.8 AND rate_p_ju <= 3.23 AND rate_skall > 16.5 THEN P(trans) = 0.27 (n = 350)
- IF rate_skall <= 8.37 AND rate_p_ju <= 6.65 AND mean_sent_len <= 21.2 AND rate_p_namligen > 3.52 THEN P(trans) = 0.11 (n = 189)
- IF rate_skall > 8.37 AND mean_sent_len > 19.8 AND rate_p_ju > 3.23 THEN P(trans) = 0.04 (n = 113)
- IF rate_skall <= 8.37 AND rate_p_ju > 6.65 AND rate_conn_T <= 8.1 AND rate_ande_part <= 8.1 THEN P(trans) = 0.03 (n = 96)
- IF rate_skall <= 8.37 AND rate_p_ju > 6.65 AND rate_conn_T <= 8.1 AND rate_ande_part > 8.1 THEN P(trans) = 0.12 (n = 95)
- IF rate_skall > 8.37 AND mean_sent_len > 19.8 AND rate_p_ju <= 3.23 AND rate_skall <= 16.5 THEN P(trans) = 0.51 (n = 857)
- IF rate_skall <= 8.37 AND rate_p_ju <= 6.65 AND mean_sent_len > 21.2 AND rate_p_namligen > 6.9 THEN P(trans) = 0.29 (n = 118)
- IF rate_skall > 8.37 AND mean_sent_len <= 19.8 AND rate_pr_ni > 8.58 THEN P(trans) = 0.43 (n = 145)

## tree_d4, gran = 500

- IF mean_sent_len > 19.7 AND rate_p_ju <= 1.91 AND rate_p_namligen <= 0.871 AND rate_skall <= 5.9 THEN P(trans) = 0.90 (n = 553)
- IF mean_sent_len <= 19.7 AND rate_skall > 3.85 AND rate_pr_hanhon <= 7 AND rate_comma > 17.8 THEN P(trans) = 0.08 (n = 513)
- IF mean_sent_len <= 19.7 AND rate_skall <= 3.85 AND rate_p_ju <= 1.92 AND rate_inte_minst <= 1.86 THEN P(trans) = 0.65 (n = 208)
- IF mean_sent_len > 19.7 AND rate_p_ju > 1.91 AND rate_poss_nonrefl <= 1.93 AND rate_kommer_att <= 2.03 THEN P(trans) = 0.00 (n = 76)
- IF mean_sent_len > 19.7 AND rate_p_ju <= 1.91 AND rate_p_namligen > 0.871 AND rate_skulle_vilja <= 1.89 THEN P(trans) = 0.24 (n = 92)
- IF mean_sent_len > 19.7 AND rate_p_ju <= 1.91 AND rate_p_namligen <= 0.871 AND rate_skall > 5.9 THEN P(trans) = 0.60 (n = 145)
- IF mean_sent_len <= 19.7 AND rate_skall <= 3.85 AND rate_p_ju > 1.92 AND rate_poss_nonrefl <= 1.93 THEN P(trans) = 0.00 (n = 40)
- IF mean_sent_len <= 19.7 AND rate_skall <= 3.85 AND rate_p_ju <= 1.92 AND rate_inte_minst > 1.86 THEN P(trans) = 0.10 (n = 40)
- IF mean_sent_len > 19.7 AND rate_p_ju > 1.91 AND rate_poss_nonrefl <= 1.93 AND rate_kommer_att > 2.03 THEN P(trans) = 0.13 (n = 23)
- IF mean_sent_len <= 19.7 AND rate_skall <= 3.85 AND rate_p_ju > 1.92 AND rate_poss_nonrefl > 1.93 THEN P(trans) = 0.10 (n = 20)
- IF mean_sent_len > 19.7 AND rate_p_ju <= 1.91 AND rate_p_namligen > 0.871 AND rate_skulle_vilja > 1.89 THEN P(trans) = 0.81 (n = 21)
- IF mean_sent_len > 19.7 AND rate_p_ju > 1.91 AND rate_poss_nonrefl > 1.93 THEN P(trans) = 0.31 (n = 29)
- IF mean_sent_len <= 19.7 AND rate_skall > 3.85 AND rate_pr_hanhon > 7 THEN P(trans) = 0.67 (n = 21)
- IF mean_sent_len <= 19.7 AND rate_skall > 3.85 AND rate_pr_hanhon <= 7 AND rate_comma <= 17.8 THEN P(trans) = 0.40 (n = 42)
