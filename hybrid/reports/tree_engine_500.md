# Depth-3 tree over the engine's rule statistics + engine score (gran=500)

Fitted on dev-train (prior P(trans)=0.47). `stat:<rule>:per1000` = hits per 1000 words, `:share_a` = share of the rule's variant A among its choices, `:mean` = unit mean.

- IF engine_score > 0.246 AND stat:starter.ska_vs_skall:choices_per1000 <= 5.81 AND stat:starter.sent_len:mean > 21.8 THEN P(trans)=0.90 (n=358)
- IF engine_score <= 0.246 AND stat:starter.ska_vs_skall:choices_per1000 > 3.84 AND engine_score <= -1.44 THEN P(trans)=0.03 (n=328)
- IF engine_score <= 0.246 AND stat:starter.ska_vs_skall:choices_per1000 > 3.84 AND engine_score > -1.44 THEN P(trans)=0.17 (n=325)
- IF engine_score > 0.246 AND stat:starter.ska_vs_skall:choices_per1000 <= 5.81 AND stat:starter.sent_len:mean <= 21.8 THEN P(trans)=0.72 (n=312)
- IF engine_score <= 0.246 AND stat:starter.ska_vs_skall:choices_per1000 <= 3.84 AND engine_score <= -1.21 THEN P(trans)=0.16 (n=124)
- IF engine_score <= 0.246 AND stat:starter.ska_vs_skall:choices_per1000 <= 3.84 AND engine_score > -1.21 THEN P(trans)=0.61 (n=177)
- IF engine_score > 0.246 AND stat:starter.ska_vs_skall:choices_per1000 > 5.81 AND engine_score > 1.97 THEN P(trans)=0.75 (n=63)
- IF engine_score > 0.246 AND stat:starter.ska_vs_skall:choices_per1000 > 5.81 AND engine_score <= 1.97 THEN P(trans)=0.43 (n=136)
