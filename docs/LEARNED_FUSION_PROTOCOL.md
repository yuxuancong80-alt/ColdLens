# 学习式 Text–Visual 晚融合协议

> 冻结日期：2026-08-14  
> 状态：训练与外层 Validation 运行前冻结  
> 最终 Test：不得读取

## 1. 目标

在不训练文本或视觉编码器的前提下，使用开发训练交互学习一个全局、可解释的内容相似度组合。该阶段用于检验：训练期评论序列能否学出比手工 rank fusion 更有效的 Text–Visual 关系。

## 2. 时序留一训练样本

1. 只读取 `split_v1/dev_train.csv.gz`。
2. 每名用户按时间排序；至少有2条交互的用户才可构造训练样本。
3. 该用户最后一次开发训练交互作为正例目标，其余更早交互组成用户画像。
4. 正例目标绝不进入该用户画像，防止目标自包含。
5. 48,422名用户满足条件；994名单交互用户不参与融合训练，但不从外层 Validation 评测用户中删除。

## 3. 负例采样

- 每个训练正例固定采样5个负例。
- 负例只来自首次观测时间不晚于正例目标时间的开发训练视频。
- 排除该用户在完整开发训练窗口内互动过的全部视频。
- 使用固定种子与用户ID生成确定性均匀采样。
- 审计中最小合法负例池为566，所有用户均满足5负例要求。

首次观测时间来自第一次公开评论，不是视频上传时间；这是可用性代理而非日志事实。均匀未交互负例可能包含未曝光或实际喜欢的视频，训练标签存在隐式反馈噪声。

## 4. 用户拆分

使用 `SHA-256("learned-fusion-v1:<user_id>")` 首字节：

- `< 26`：内部 calibration 用户，约10%；
- 其他：内部 train 用户，约90%。

同一用户的配对不会跨集合。内部 calibration 只选择训练轮数，不搜索特征、模型宽度或负例数。

## 5. 输入特征与模型

对正例和每个负例分别计算：

1. `text_cosine`：历史标题 TF-IDF 用户画像与候选标题的余弦相似度；
2. `visual_cosine`：历史封面 MobileNetV3-Small 用户画像与候选封面的余弦相似度；
3. `text_cosine × visual_cosine`。

TF-IDF 的 IDF 只用开发训练视频拟合；视觉编码器保持冻结。三个特征使用内部 train 配对中的候选特征均值和标准差进行标准化。

模型只有三个可学习 logits，经 softmax 转换成非负且和为1的权重：

`score = w_text × z_text + w_visual × z_visual + w_interaction × z_interaction`

模型不使用用户ID、视频ID、全局点赞、播放量、候选未来交互或测试信息。

## 6. 冻结训练预算

- 损失：pairwise logistic，`softplus(-(score_positive - score_negative))`；
- 优化器：Adam；
- 学习率：0.01；
- batch size：1,024；
- 最大轮数：50；
- calibration patience：5；
- 最小改善：`1e-6`；
- 随机种子：20260814；
- 设备：本地 CPU；
- 选择：calibration pairwise loss 最低轮数；
- 选定轮数后，从相同初始化在全部48,422名合格用户配对上重训，不再早停。

## 7. 外层 Validation 规则

- 外层用户画像使用该用户完整 `dev_train` 历史；Validation 正例不进入画像。
- 对完整1,305个 Validation Cold 候选排序。
- 报告与既有基线相同的 Recall/NDCG/HitRate @5/10/20/50。
- 同时按冻结的 `visual_clean_dhash0` 候选与用户口径报告敏感性结果，不重新训练或调权。
- 只运行一个冻结模型版本，不根据外层 Validation 结果追加特征或超参数搜索。

## 8. 接受标准

学习式融合只有在主 Validation 的 NDCG@10 与 Recall@10 都严格超过 Text-only 时，才作为最终候选；否则保留 Text-only。无论结果如何都记录完整结果，不运行新的启发式补救搜索。

在完成外层 Validation、代码复现与模型冻结记录前，不运行最终 Test 模型指标。
