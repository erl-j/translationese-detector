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
| 20261105T223446Z | main | 69eb2eee | `tsw.engine:Engine(exclude="starter.sent_len")` | dev | sent | bal | - | 87904 | 0.4842 | 0.4487 | 0.5394 |
| 20261105T223447Z | main | 69eb2eee+dirty | `tsw.engine:Default` | dev | sent | bal | - | 87904 | 0.5601 | 0.5598 | 0.5867 |
| 20261105T223457Z | main | 69eb2eee+dirty | `tsw.engine:Engine(exclude="starter.sent_len")` | dev | sent | bal | masked | 87904 | 0.4842 | 0.4487 | 0.5392 |
| 20261105T223459Z | main | 69eb2eee+dirty | `tsw.engine:Default` | dev | sent | bal | masked | 87904 | 0.5598 | 0.5596 | 0.5865 |
| 20261105T223504Z | main | 69eb2eee+dirty | `tsw.engine:Engine(exclude="starter.sent_len")` | dev | 100 | bal | - | 14170 | 0.5190 | 0.4870 | 0.5676 |
| 20261105T223506Z | main | 69eb2eee+dirty | `tsw.engine:Default` | dev | 100 | bal | - | 14170 | 0.6494 | 0.6474 | 0.7109 |
| 20261105T223513Z | main | 69eb2eee+dirty | `tsw.engine:Engine(exclude="starter.sent_len")` | dev | 100 | bal | masked | 14170 | 0.5189 | 0.4869 | 0.5673 |
| 20261105T223515Z | main | 69eb2eee+dirty | `tsw.engine:Default` | dev | 100 | bal | masked | 14170 | 0.6491 | 0.6472 | 0.7104 |
| 20261105T223518Z | main | 69eb2eee+dirty | `tsw.engine:Engine(exclude="starter.sent_len")` | dev | 500 | bal | - | 2734 | 0.6335 | 0.6173 | 0.7084 |
| 20261105T223520Z | main | 69eb2eee+dirty | `tsw.engine:Default` | dev | 500 | bal | - | 2734 | 0.7714 | 0.7705 | 0.8473 |
| 20261105T223526Z | main | 69eb2eee+dirty | `tsw.engine:Engine(exclude="starter.sent_len")` | dev | 500 | bal | masked | 2734 | 0.6335 | 0.6173 | 0.7082 |
| 20261105T223528Z | main | 69eb2eee+dirty | `tsw.engine:Default` | dev | 500 | bal | masked | 2734 | 0.7714 | 0.7705 | 0.8468 |
| 20261105T224924Z | feat/register | 9033fde8 | `tsw.engine:Default` | dev | sent | bal | - | 87904 | 0.6037 | 0.6032 | 0.6521 |
| 20261105T224941Z | feat/register | 9033fde8+dirty | `tsw.engine:Default` | dev | sent | bal | masked | 87904 | 0.6036 | 0.6032 | 0.6520 |
| 20261105T224951Z | feat/register | 9033fde8+dirty | `tsw.engine:Default` | dev | 100 | bal | - | 14170 | 0.7348 | 0.7344 | 0.8108 |
| 20261105T225004Z | feat/register | 9033fde8+dirty | `tsw.engine:Default` | dev | 100 | bal | masked | 14170 | 0.7347 | 0.7343 | 0.8107 |
| 20261105T225012Z | feat/register | 9033fde8+dirty | `tsw.engine:Default` | dev | 500 | bal | - | 2734 | 0.8749 | 0.8747 | 0.9468 |
| 20261105T225022Z | feat/register | 9033fde8+dirty | `tsw.engine:Default` | dev | 500 | bal | masked | 2734 | 0.8745 | 0.8743 | 0.9467 |
| 20261105T225042Z | feat/register | 9033fde8+dirty | `tsw.engine:Families(only="register")` | dev | sent | bal | - | 87904 | 0.5831 | 0.5682 | 0.6075 |
| 20261105T225055Z | feat/register | 9033fde8+dirty | `tsw.engine:Families(only="register")` | dev | sent | bal | masked | 87904 | 0.5831 | 0.5682 | 0.6076 |
| 20261105T225103Z | feat/register | 9033fde8+dirty | `tsw.engine:Families(only="register")` | dev | 100 | bal | - | 14170 | 0.7020 | 0.7017 | 0.7712 |
| 20261105T225113Z | feat/register | 9033fde8+dirty | `tsw.engine:Families(only="register")` | dev | 100 | bal | masked | 14170 | 0.7020 | 0.7017 | 0.7713 |
| 20261105T225120Z | feat/register | 9033fde8+dirty | `tsw.engine:Families(only="register")` | dev | 500 | bal | - | 2734 | 0.8376 | 0.8376 | 0.9192 |
| 20261105T225128Z | feat/register | 9033fde8+dirty | `tsw.engine:Families(only="register")` | dev | 500 | bal | masked | 2734 | 0.8376 | 0.8376 | 0.9193 |
