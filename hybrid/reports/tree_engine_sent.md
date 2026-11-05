# Depth-3 tree over the engine's rule statistics + engine score (gran=sent)

Fitted on dev-train (prior P(trans)=0.47). `stat:<rule>:per1000` = hits per 1000 words, `:share_a` = share of the rule's variant A among its choices, `:mean` = unit mean.

- IF engine_score > 0.139 AND stat:starter.ska_vs_skall:choices_per1000 <= 19.4 AND engine_score <= 0.339 THEN P(trans)=0.61 (n=6803)
- IF engine_score > 0.139 AND stat:starter.ska_vs_skall:choices_per1000 <= 19.4 AND engine_score > 0.339 THEN P(trans)=0.77 (n=2523)
- IF engine_score <= 0.139 AND stat:starter.ska_vs_skall:share_a > 0.375 AND engine_score > -1.34 THEN P(trans)=0.46 (n=43769)
- IF engine_score <= 0.139 AND stat:starter.ska_vs_skall:share_a <= 0.375 AND stat:starter.ska_vs_skall:choices_per1000 > 48.8 THEN P(trans)=0.23 (n=2122)
- IF engine_score <= 0.139 AND stat:starter.ska_vs_skall:share_a > 0.375 AND engine_score <= -1.34 THEN P(trans)=0.16 (n=906)
- IF engine_score <= 0.139 AND stat:starter.ska_vs_skall:share_a <= 0.375 AND stat:starter.ska_vs_skall:choices_per1000 <= 48.8 THEN P(trans)=0.31 (n=1371)
- IF engine_score > 0.139 AND stat:starter.ska_vs_skall:choices_per1000 > 19.4 AND engine_score <= 0.228 THEN P(trans)=0.36 (n=605)
- IF engine_score > 0.139 AND stat:starter.ska_vs_skall:choices_per1000 > 19.4 AND engine_score > 0.228 THEN P(trans)=0.46 (n=622)
