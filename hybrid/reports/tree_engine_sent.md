# Depth-3 tree over the engine's rule statistics + engine score (gran=sent)

Fitted on trn (prior P(trans)=0.50). `stat:<rule>:per1000` = hits per 1000 words, `:share_a` = share of the rule's variant A among its choices, `:mean` = unit mean. Engine `tsw.engine:Default`, tsw hash `00a590c8e3449552`.

- IF engine_score <= 0.0467 AND engine_score <= -0.436 AND engine_score > -1.39 THEN P(trans)=0.26 (n=13328)
- IF engine_score > 0.0467 AND engine_score <= 0.887 AND engine_score > 0.225 THEN P(trans)=0.65 (n=15263)
- IF engine_score > 0.0467 AND engine_score > 0.887 AND engine_score <= 1.52 THEN P(trans)=0.78 (n=7248)
- IF engine_score <= 0.0467 AND engine_score <= -0.436 AND engine_score <= -1.39 THEN P(trans)=0.09 (n=4912)
- IF engine_score > 0.0467 AND engine_score > 0.887 AND engine_score > 1.52 THEN P(trans)=0.90 (n=4943)
- IF engine_score <= 0.0467 AND engine_score > -0.436 AND engine_score > -0.289 THEN P(trans)=0.46 (n=27177)
- IF engine_score > 0.0467 AND engine_score <= 0.887 AND engine_score <= 0.225 THEN P(trans)=0.56 (n=10842)
- IF engine_score <= 0.0467 AND engine_score > -0.436 AND engine_score <= -0.289 THEN P(trans)=0.37 (n=4191)
