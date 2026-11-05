# Leaderboard

One line per `evals/run.py` invocation (appended automatically). AUC is ROC-AUC of the raw scores; `mask` = topic tokens masked.

| time (UTC) | branch | commit | clf | split | gran | data | mask | n | acc | macro-F1 | AUC |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|
| 20261105T221720Z | main | e35cd869 | `baselines:Constant` | dev | sent | bal | - | 87904 | 0.5000 | 0.3333 | 0.5000 |
| 20261105T221724Z | main | e35cd869 | `baselines:Constant` | dev | sent | bal | masked | 87904 | 0.5000 | 0.3333 | 0.5000 |
| 20261105T221724Z | main | e35cd869 | `baselines:Constant` | dev | 100 | bal | - | 14170 | 0.5000 | 0.3333 | 0.5000 |
| 20261105T221727Z | main | e35cd869 | `baselines:Constant` | dev | 100 | bal | masked | 14170 | 0.5000 | 0.3333 | 0.5000 |
| 20261105T221728Z | main | e35cd869 | `baselines:Constant` | dev | 500 | bal | - | 2734 | 0.5000 | 0.3333 | 0.5000 |
| 20261105T221730Z | main | e35cd869 | `baselines:Constant` | dev | 500 | bal | masked | 2734 | 0.5000 | 0.3333 | 0.5000 |
| 20261105T221731Z | main | e35cd869 | `baselines:Random(seed=0)` | dev | sent | bal | - | 87904 | 0.4993 | 0.4993 | 0.5003 |
| 20261105T221735Z | main | e35cd869 | `baselines:Random(seed=0)` | dev | sent | bal | masked | 87904 | 0.4995 | 0.4995 | 0.5005 |
| 20261105T221735Z | main | e35cd869 | `baselines:Random(seed=0)` | dev | 100 | bal | - | 14170 | 0.5016 | 0.5015 | 0.4987 |
| 20261105T221739Z | main | e35cd869 | `baselines:Random(seed=0)` | dev | 100 | bal | masked | 14170 | 0.5001 | 0.5001 | 0.5000 |
| 20261105T221739Z | main | e35cd869 | `baselines:Random(seed=0)` | dev | 500 | bal | - | 2734 | 0.4770 | 0.4769 | 0.4775 |
| 20261105T221741Z | main | e35cd869 | `baselines:Random(seed=0)` | dev | 500 | bal | masked | 2734 | 0.4909 | 0.4908 | 0.4930 |
| 20261105T221742Z | main | e35cd869 | `baselines:MeanSentLen` | dev | sent | bal | - | 87904 | 0.5614 | 0.5597 | 0.5826 |
| 20261105T221747Z | main | e35cd869 | `baselines:MeanSentLen` | dev | sent | bal | masked | 87904 | 0.5613 | 0.5596 | 0.5824 |
| 20261105T221747Z | main | e35cd869 | `baselines:MeanSentLen` | dev | 100 | bal | - | 14170 | 0.6291 | 0.6291 | 0.6771 |
| 20261105T221751Z | main | e35cd869 | `baselines:MeanSentLen` | dev | 100 | bal | masked | 14170 | 0.6289 | 0.6289 | 0.6766 |
| 20261105T221751Z | main | e35cd869 | `baselines:MeanSentLen` | dev | 500 | bal | - | 2734 | 0.7217 | 0.7212 | 0.7879 |
| 20261105T221754Z | main | e35cd869 | `baselines:MeanSentLen` | dev | 500 | bal | masked | 2734 | 0.7209 | 0.7205 | 0.7872 |
| 20261105T221755Z | main | e35cd869 | `baselines:TopicCheat` | dev | sent | bal | - | 87904 | 0.5160 | 0.3727 | 0.5358 |
| 20261105T221800Z | main | e35cd869 | `baselines:TopicCheat` | dev | sent | bal | masked | 87904 | 0.5000 | 0.3333 | 0.5000 |
| 20261105T221801Z | main | e35cd869 | `baselines:TopicCheat` | dev | 100 | bal | - | 14170 | 0.5569 | 0.4634 | 0.6318 |
| 20261105T221805Z | main | e35cd869 | `baselines:TopicCheat` | dev | 100 | bal | masked | 14170 | 0.5000 | 0.3333 | 0.5000 |
| 20261105T221805Z | main | e35cd869 | `baselines:TopicCheat` | dev | 500 | bal | - | 2734 | 0.6609 | 0.6331 | 0.7994 |
| 20261105T221808Z | main | e35cd869 | `baselines:TopicCheat` | dev | 500 | bal | masked | 2734 | 0.5000 | 0.3333 | 0.5000 |
