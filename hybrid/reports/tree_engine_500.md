# Depth-3 tree over the engine's rule statistics + engine score (gran=500)

Fitted on trn (prior P(trans)=0.50). `stat:<rule>:per1000` = hits per 1000 words, `:share_a` = share of the rule's variant A among its choices, `:mean` = unit mean. Engine `tsw.engine:Default`, tsw hash `263693870a139a72`.

- IF engine_score > 0.032 AND stat:starter.ska_vs_skall:choices_per1000 <= 5.81 AND engine_score > 0.822 THEN P(trans)=0.89 (n=801)
- IF engine_score <= 0.032 AND stat:starter.ska_vs_skall:choices_per1000 > 1.97 AND engine_score <= -0.764 THEN P(trans)=0.06 (n=642)
- IF engine_score <= 0.032 AND stat:starter.ska_vs_skall:choices_per1000 > 1.97 AND engine_score > -0.764 THEN P(trans)=0.23 (n=314)
- IF engine_score > 0.032 AND stat:starter.ska_vs_skall:choices_per1000 <= 5.81 AND engine_score <= 0.822 THEN P(trans)=0.68 (n=350)
- IF engine_score <= 0.032 AND stat:starter.ska_vs_skall:choices_per1000 <= 1.97 AND engine_score <= -1.49 THEN P(trans)=0.19 (n=99)
- IF engine_score > 0.032 AND stat:starter.ska_vs_skall:choices_per1000 > 5.81 AND engine_score > 1.72 THEN P(trans)=0.76 (n=110)
- IF engine_score > 0.032 AND stat:starter.ska_vs_skall:choices_per1000 > 5.81 AND engine_score <= 1.72 THEN P(trans)=0.40 (n=255)
- IF engine_score <= 0.032 AND stat:starter.ska_vs_skall:choices_per1000 <= 1.97 AND engine_score > -1.49 THEN P(trans)=0.63 (n=163)
