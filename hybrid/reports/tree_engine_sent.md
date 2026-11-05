# Depth-3 tree over the engine's rule statistics + engine score (gran=sent)

Fitted on trn (prior P(trans)=0.50). `stat:<rule>:per1000` = hits per 1000 words, `:share_a` = share of the rule's variant A among its choices, `:mean` = unit mean. Engine `tsw.engine:Default`, tsw hash `263693870a139a72`.

- IF engine_score > 0.139 AND stat:starter.ska_vs_skall:choices_per1000 <= 19.4 AND engine_score <= 0.34 THEN P(trans)=0.64 (n=10352)
- IF engine_score <= 0.139 AND stat:starter.ska_vs_skall:choices_per1000 <= 39.2 AND engine_score > -1.46 THEN P(trans)=0.48 (n=66647)
- IF engine_score > 0.139 AND stat:starter.ska_vs_skall:choices_per1000 <= 19.4 AND engine_score > 0.34 THEN P(trans)=0.80 (n=3714)
- IF engine_score <= 0.139 AND stat:starter.ska_vs_skall:choices_per1000 > 39.2 AND stat:starter.ska_vs_skall:choices_per1000 > 56.3 THEN P(trans)=0.24 (n=2345)
- IF engine_score <= 0.139 AND stat:starter.ska_vs_skall:choices_per1000 > 39.2 AND stat:starter.ska_vs_skall:choices_per1000 <= 56.3 THEN P(trans)=0.30 (n=1914)
- IF engine_score <= 0.139 AND stat:starter.ska_vs_skall:choices_per1000 <= 39.2 AND engine_score <= -1.46 THEN P(trans)=0.18 (n=1121)
- IF engine_score > 0.139 AND stat:starter.ska_vs_skall:choices_per1000 > 19.4 AND engine_score <= 0.228 THEN P(trans)=0.39 (n=900)
- IF engine_score > 0.139 AND stat:starter.ska_vs_skall:choices_per1000 > 19.4 AND engine_score > 0.228 THEN P(trans)=0.50 (n=911)
