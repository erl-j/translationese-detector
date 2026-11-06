# Depth-3 tree over the engine's rule statistics + engine score (gran=100)

Fitted on trn (prior P(trans)=0.50). `stat:<rule>:per1000` = hits per 1000 words, `:share_a` = share of the rule's variant A among its choices, `:mean` = unit mean. Engine `tsw.engine:Default`, tsw hash `00a590c8e3449552`.

- IF engine_score <= -0.147 AND engine_score <= -1.3 AND engine_score <= -2.5 THEN P(trans)=0.02 (n=2724)
- IF engine_score > -0.147 AND engine_score > 1.07 AND engine_score > 2.45 THEN P(trans)=0.95 (n=2836)
- IF engine_score > -0.147 AND engine_score > 1.07 AND engine_score <= 2.45 THEN P(trans)=0.82 (n=2560)
- IF engine_score <= -0.147 AND engine_score <= -1.3 AND engine_score > -2.5 THEN P(trans)=0.13 (n=1601)
- IF engine_score <= -0.147 AND engine_score > -1.3 AND engine_score <= -0.591 THEN P(trans)=0.26 (n=1201)
- IF engine_score > -0.147 AND engine_score <= 1.07 AND stat:connpunct.native_connectives:per1000 <= 7.6 THEN P(trans)=0.62 (n=1735)
- IF engine_score <= -0.147 AND engine_score > -1.3 AND engine_score > -0.591 THEN P(trans)=0.39 (n=876)
- IF engine_score > -0.147 AND engine_score <= 1.07 AND stat:connpunct.native_connectives:per1000 > 7.6 THEN P(trans)=0.46 (n=637)
