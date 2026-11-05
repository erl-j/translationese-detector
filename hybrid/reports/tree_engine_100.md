# Depth-3 tree over the engine's rule statistics + engine score (gran=100)

Fitted on dev-train (prior P(trans)=0.47). `stat:<rule>:per1000` = hits per 1000 words, `:share_a` = share of the rule's variant A among its choices, `:mean` = unit mean.

- IF engine_score > 0.0552 AND stat:starter.ska_vs_skall:choices_per1000 <= 8.58 AND engine_score <= 0.578 THEN P(trans)=0.62 (n=2718)
- IF engine_score > 0.0552 AND stat:starter.ska_vs_skall:choices_per1000 <= 8.58 AND engine_score > 0.578 THEN P(trans)=0.81 (n=1068)
- IF engine_score <= 0.0552 AND stat:starter.ska_vs_skall:share_a <= 0.375 AND engine_score > -1.15 THEN P(trans)=0.23 (n=1459)
- IF engine_score <= 0.0552 AND stat:starter.ska_vs_skall:share_a <= 0.375 AND engine_score <= -1.15 THEN P(trans)=0.05 (n=387)
- IF engine_score <= 0.0552 AND stat:starter.ska_vs_skall:share_a > 0.375 AND engine_score <= -1.28 THEN P(trans)=0.18 (n=448)
- IF engine_score > 0.0552 AND stat:starter.ska_vs_skall:choices_per1000 > 8.58 AND stat:starter.ska_vs_skall:choices_per1000 > 15.7 THEN P(trans)=0.27 (n=390)
- IF engine_score <= 0.0552 AND stat:starter.ska_vs_skall:share_a > 0.375 AND engine_score > -1.28 THEN P(trans)=0.45 (n=2328)
- IF engine_score > 0.0552 AND stat:starter.ska_vs_skall:choices_per1000 > 8.58 AND stat:starter.ska_vs_skall:choices_per1000 <= 15.7 THEN P(trans)=0.50 (n=762)
