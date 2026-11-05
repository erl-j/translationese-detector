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
| 20261105T230011Z | feat/register | 48051092 | `tsw.engine:Default` | dev | sent | bal | - | 87904 | 0.6040 | 0.6035 | 0.6522 |
| 20261105T230028Z | feat/register | 48051092+dirty | `tsw.engine:Default` | dev | sent | bal | masked | 87904 | 0.6039 | 0.6034 | 0.6521 |
| 20261105T230038Z | feat/register | 48051092+dirty | `tsw.engine:Default` | dev | 100 | bal | - | 14170 | 0.7354 | 0.7349 | 0.8105 |
| 20261105T230050Z | feat/register | 48051092+dirty | `tsw.engine:Default` | dev | 100 | bal | masked | 14170 | 0.7354 | 0.7348 | 0.8104 |
| 20261105T230058Z | feat/register | 48051092+dirty | `tsw.engine:Default` | dev | 500 | bal | - | 2734 | 0.8756 | 0.8753 | 0.9467 |
| 20261105T230108Z | feat/register | 48051092+dirty | `tsw.engine:Default` | dev | 500 | bal | masked | 2734 | 0.8753 | 0.8749 | 0.9466 |
| 20261105T230124Z | feat/register | 48051092+dirty | `tsw.engine:Families(only="register")` | dev | sent | bal | - | 87904 | 0.5830 | 0.5679 | 0.6073 |
| 20261105T230137Z | feat/register | 48051092+dirty | `tsw.engine:Families(only="register")` | dev | sent | bal | masked | 87904 | 0.5830 | 0.5679 | 0.6073 |
| 20261105T230145Z | feat/register | 48051092+dirty | `tsw.engine:Families(only="register")` | dev | 100 | bal | - | 14170 | 0.7040 | 0.7037 | 0.7701 |
| 20261105T230155Z | feat/register | 48051092+dirty | `tsw.engine:Families(only="register")` | dev | 100 | bal | masked | 14170 | 0.7040 | 0.7037 | 0.7701 |
| 20261105T230201Z | feat/register | 48051092+dirty | `tsw.engine:Families(only="register")` | dev | 500 | bal | - | 2734 | 0.8376 | 0.8375 | 0.9185 |
| 20261105T230209Z | feat/register | 48051092+dirty | `tsw.engine:Families(only="register")` | dev | 500 | bal | masked | 2734 | 0.8376 | 0.8375 | 0.9185 |
| 20261105T230300Z | feat/register | f7c1f3d5 | `tsw.engine:Default` | val | sent | bal | - | 21728 | 0.5867 | 0.5859 | 0.6235 |
| 20261105T230304Z | feat/register | f7c1f3d5+dirty | `tsw.engine:Default` | val | sent | bal | masked | 21728 | 0.5865 | 0.5857 | 0.6233 |
| 20261105T230307Z | feat/register | f7c1f3d5+dirty | `tsw.engine:Default` | val | 100 | bal | - | 3354 | 0.6911 | 0.6907 | 0.7543 |
| 20261105T230310Z | feat/register | f7c1f3d5+dirty | `tsw.engine:Default` | val | 100 | bal | masked | 3354 | 0.6908 | 0.6904 | 0.7542 |
| 20261105T230312Z | feat/register | f7c1f3d5+dirty | `tsw.engine:Default` | val | 500 | bal | - | 636 | 0.8223 | 0.8220 | 0.9134 |
| 20261105T230314Z | feat/register | f7c1f3d5+dirty | `tsw.engine:Default` | val | 500 | bal | masked | 636 | 0.8223 | 0.8220 | 0.9133 |
| 20261105T230317Z | feat/register | f7c1f3d5+dirty | `tsw.engine:Families(only="register")` | val | sent | bal | - | 21728 | 0.5622 | 0.5444 | 0.5716 |
| 20261105T230320Z | feat/register | f7c1f3d5+dirty | `tsw.engine:Families(only="register")` | val | sent | bal | masked | 21728 | 0.5622 | 0.5444 | 0.5717 |
| 20261105T230322Z | feat/register | f7c1f3d5+dirty | `tsw.engine:Families(only="register")` | val | 100 | bal | - | 3354 | 0.6592 | 0.6589 | 0.7182 |
| 20261105T230325Z | feat/register | f7c1f3d5+dirty | `tsw.engine:Families(only="register")` | val | 100 | bal | masked | 3354 | 0.6592 | 0.6589 | 0.7183 |
| 20261105T230326Z | feat/register | f7c1f3d5+dirty | `tsw.engine:Families(only="register")` | val | 500 | bal | - | 636 | 0.7830 | 0.7827 | 0.8700 |
| 20261105T230328Z | feat/register | f7c1f3d5+dirty | `tsw.engine:Families(only="register")` | val | 500 | bal | masked | 636 | 0.7830 | 0.7827 | 0.8701 |
| 20261105T230331Z | feat/register | f7c1f3d5+dirty | `tsw.engine:Engine(include_disabled=True, exclude="register,starter.formal_connectives")` | val | sent | bal | - | 21728 | 0.5598 | 0.5593 | 0.5767 |
| 20261105T230334Z | feat/register | f7c1f3d5+dirty | `tsw.engine:Engine(include_disabled=True, exclude="register,starter.formal_connectives")` | val | sent | bal | masked | 21728 | 0.5597 | 0.5592 | 0.5765 |
| 20261105T230335Z | feat/register | f7c1f3d5+dirty | `tsw.engine:Engine(include_disabled=True, exclude="register,starter.formal_connectives")` | val | 100 | bal | - | 3354 | 0.6109 | 0.6073 | 0.6403 |
| 20261105T230338Z | feat/register | f7c1f3d5+dirty | `tsw.engine:Engine(include_disabled=True, exclude="register,starter.formal_connectives")` | val | 100 | bal | masked | 3354 | 0.6109 | 0.6073 | 0.6401 |
| 20261105T230339Z | feat/register | f7c1f3d5+dirty | `tsw.engine:Engine(include_disabled=True, exclude="register,starter.formal_connectives")` | val | 500 | bal | - | 636 | 0.6997 | 0.6955 | 0.7906 |
| 20261105T230341Z | feat/register | f7c1f3d5+dirty | `tsw.engine:Engine(include_disabled=True, exclude="register,starter.formal_connectives")` | val | 500 | bal | masked | 636 | 0.6997 | 0.6955 | 0.7904 |
