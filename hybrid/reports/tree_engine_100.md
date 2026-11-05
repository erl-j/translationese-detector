# Depth-3 tree over the engine's rule statistics + engine score (gran=100)

Fitted on trn (prior P(trans)=0.50). `stat:<rule>:per1000` = hits per 1000 words, `:share_a` = share of the rule's variant A among its choices, `:mean` = unit mean. Engine `tsw.engine:Default`, tsw hash `263693870a139a72`.

- IF engine_score > 0.0182 AND stat:starter.ska_vs_skall:choices_per1000 <= 8.37 AND engine_score > 0.426 THEN P(trans)=0.80 (n=2413)
- IF engine_score > 0.0182 AND stat:starter.ska_vs_skall:choices_per1000 <= 8.37 AND engine_score <= 0.426 THEN P(trans)=0.63 (n=3540)
- IF engine_score <= 0.0182 AND stat:starter.ska_vs_skall:choices_per1000 > 6.57 AND engine_score <= -0.304 THEN P(trans)=0.13 (n=1247)
- IF engine_score <= 0.0182 AND stat:starter.ska_vs_skall:choices_per1000 <= 6.57 AND engine_score <= -0.352 THEN P(trans)=0.31 (n=1611)
- IF engine_score <= 0.0182 AND stat:starter.ska_vs_skall:choices_per1000 > 6.57 AND engine_score > -0.304 THEN P(trans)=0.26 (n=1176)
- IF engine_score > 0.0182 AND stat:starter.ska_vs_skall:choices_per1000 > 8.37 AND stat:starter.ska_vs_skall:choices_per1000 > 15.7 THEN P(trans)=0.29 (n=646)
- IF engine_score > 0.0182 AND stat:starter.ska_vs_skall:choices_per1000 > 8.37 AND stat:starter.ska_vs_skall:choices_per1000 <= 15.7 THEN P(trans)=0.52 (n=1362)
- IF engine_score <= 0.0182 AND stat:starter.ska_vs_skall:choices_per1000 <= 6.57 AND engine_score > -0.352 THEN P(trans)=0.49 (n=2175)
