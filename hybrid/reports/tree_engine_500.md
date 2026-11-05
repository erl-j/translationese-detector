# Depth-3 tree over the engine's rule statistics + engine score (gran=500)

Fitted on trn (prior P(trans)=0.50). `stat:<rule>:per1000` = hits per 1000 words, `:share_a` = share of the rule's variant A among its choices, `:mean` = unit mean. Engine `tsw.engine:Default`, tsw hash `00a590c8e3449552`.

- IF engine_score <= 1.91 AND engine_score <= -0.525 AND engine_score <= -2.74 THEN P(trans)=0.00 (n=1102)
- IF engine_score > 1.91 AND engine_score > 3.81 AND engine_score > 5.2 THEN P(trans)=0.99 (n=1057)
- IF engine_score > 1.91 AND engine_score > 3.81 AND engine_score <= 5.2 THEN P(trans)=0.92 (n=146)
- IF engine_score <= 1.91 AND engine_score <= -0.525 AND engine_score > -2.74 THEN P(trans)=0.14 (n=147)
- IF engine_score > 1.91 AND engine_score <= 3.81 AND stat:register.tycker_vs_anser:share_a > 0.438 THEN P(trans)=0.86 (n=96)
- IF engine_score <= 1.91 AND engine_score > -0.525 AND stat:connpunct.additive_ocksa:per1000 > 2.01 THEN P(trans)=0.24 (n=68)
- IF engine_score <= 1.91 AND engine_score > -0.525 AND stat:connpunct.additive_ocksa:per1000 <= 2.01 THEN P(trans)=0.56 (n=75)
- IF engine_score > 1.91 AND engine_score <= 3.81 AND stat:register.tycker_vs_anser:share_a <= 0.438 THEN P(trans)=0.56 (n=43)
