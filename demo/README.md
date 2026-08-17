# ColdLens Local Demo

这是 ColdLens 的本地解释型 Demo，包含三个主模式：

- 8 个确定性选择的匿名 Validation 用户，以及 Text-only TF-IDF 的真实离线 Top-10 和匹配词；
- 用户选择 1–6 条训练期历史视频，按正式实验同一画像公式对完整 1,305 个 Validation 冷候选实时排序，并查看每条推荐主要关联哪段历史；
- 也可输入英文兴趣词做快速近似体验。
- 方案对比集中展示冻结的 Text/Visual Test 汇总、三类融合在 Validation 被拒绝的原因和最终产品取舍。

Demo 不训练模型，也不在运行时读取 Test 标签或报告文件；方案对比只显示已写入正式文档的冻结汇总数字。余弦相似度不显示成概率，模拟模式没有未来行为标签且不计算准确率。

## 本地启动

先在项目根目录生成匿名样例：

```powershell
.\.venv\Scripts\python.exe scripts\export_demo_data.py
```

再启动页面：

```powershell
Set-Location demo
npm.cmd ci
npm.cmd run dev
```

浏览器打开 `http://localhost:3000/`。

已安装依赖时可以跳过 `npm.cmd ci`。Node.js 要求 22.13 或更高版本。

## 数据边界

- 本地生成文件：`public/demo-data.json`
- 文件被父项目 `.gitignore` 排除，不随仓库发布
- 不包含原始用户 ID、完整交互表、封面或特征矩阵
- 每个历史分层固定展示一个命中和一个未命中样例，不代表真实命中率
- 模拟输入和排序完全在浏览器中完成，不保存、不调用 API
- 历史演示目录是 6 个主题各 60 条、共 360 条的固定平衡样本，不代表真实目录分布
- 兴趣词模式提供 6 类共 58 个经过模型词表核验的可点击词，仍允许自由输入其他英文/数字词
- “主要相关历史”是文本相似度解释，不是用户偏好的因果归因
- 当前冻结 tokenizer 只可靠识别英文和数字，不具备中文语义理解
- 方案对比不为 Validation 已拒绝的融合补算 Test，也不把离线差值解释成线上收益

详细规则见 [`docs/DEMO_PROTOCOL.md`](../docs/DEMO_PROTOCOL.md)。
面试展示时可按 [`docs/INTERVIEW_WALKTHROUGH.md`](../docs/INTERVIEW_WALKTHROUGH.md) 的顺序操作三个模式。

## 验证

```powershell
npm.cmd run build
node --test tests\rendered-html.test.mjs
```
