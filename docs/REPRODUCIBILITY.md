# ColdLens 复现指南

## 1. 已验证环境

- Windows / PowerShell
- Python 3.12.10
- NumPy 2.5.2
- Pillow 12.3.0
- PyTorch 2.10.0+cpu
- Torchvision 0.25.0+cpu
- 全流程可在 CPU 上运行；本项目未使用公司服务器

建议将仓库和虚拟环境放在空间充足的非系统盘。Python 脚本通过自身位置解析项目根目录，因此不要求仓库固定在 `D:\ColdLens`；两个 PowerShell 封面工具的默认路径是该目录，如位置不同请显式传参。

## 2. 数据边界

数据不随仓库发布。请先阅读 [MicroLens 官方仓库](https://github.com/westlake-repl/MicroLens) 与[官方数据门户](https://recsys.westlake.edu.cn/)的获取和使用说明。

期望的本地结构：

```text
data/raw/microlens50k/
├── MicroLens-50k_pairs.csv
├── MicroLens-50k_titles.csv
└── MicroLens-50k_covers.zip
```

冻结配置中的元数据校验值：

- pairs SHA-256：`7ff8b91bcc84f5434ac2c5be7d0b7d7730f5e84f79f9648b5ae67a7641f97bbd`
- titles SHA-256：`244aad5380cbbe0fb43458cfcda5ebe820f534384602f80a64dbbcd07dd30e49`
- covers ZIP SHA-256：`135255149cc74d47f1fb04985b2d16b862ee3d00c963ff47cb756b2136ca5892`

不要提交或再分发原始数据、封面、特征与实验输出；这些目录已在 `.gitignore` 中排除。

## 3. 环境安装

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements.txt` 固定的是本项目实际验证版本。PyTorch CPU wheel 通过官方索引获取，不需要 CUDA。

## 4. 从审计到 Validation

在仓库根目录依次运行：

```powershell
.\.venv\Scripts\python.exe scripts\audit_microlens50k.py
.\.venv\Scripts\python.exe scripts\build_split_v1.py
.\.venv\Scripts\python.exe scripts\evaluate_random_cold.py
.\.venv\Scripts\python.exe scripts\run_text_tfidf_validation.py
.\.venv\Scripts\python.exe scripts\analyze_text_tfidf_validation.py
```

封面审计与视觉特征：

```powershell
.\.venv\Scripts\python.exe scripts\audit_microlens50k_covers.py
powershell -ExecutionPolicy Bypass -File scripts\extract_cover_dhash.ps1
.\.venv\Scripts\python.exe scripts\audit_cover_dhash_boundaries.py
.\.venv\Scripts\python.exe scripts\run_mobilenet_v3_small_smoke.py
.\.venv\Scripts\python.exe scripts\extract_mobilenet_v3_small_features.py
.\.venv\Scripts\python.exe scripts\run_visual_mobilenet_validation.py
```

视觉权重首次使用时需要下载 torchvision 官方预训练权重；若要求完全离线，请提前把对应权重放入 PyTorch 本地缓存。原项目的全量 19,220 张封面 CPU 特征提取耗时 235.49 秒，但机器、缓存与存储不同会导致运行时间变化。

融合实验：

```powershell
.\.venv\Scripts\python.exe scripts\run_multimodal_rank_fusion_validation.py
.\.venv\Scripts\python.exe scripts\run_conditional_modality_gate_validation.py
.\.venv\Scripts\python.exe scripts\build_learned_fusion_pairs.py
.\.venv\Scripts\python.exe scripts\train_learned_fusion.py
.\.venv\Scripts\python.exe scripts\evaluate_learned_fusion_validation.py
```

冻结后的数据与产品诊断（只读取全量描述字段和 Validation，不读取 Test）：

```powershell
.\.venv\Scripts\python.exe scripts\analyze_product_metrics.py
```

## 5. 最终测试边界

先用预检入口确认开发期端点可复现；该模式不读取最终训练或 Test 文件：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_frozen_final_test.py --preflight-only
```

完整最终测试入口是：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_frozen_final_test.py
```

本项目的正式 Test 已于 2026-08-14 完成并锁定。若只是继续开发新模型，**不要运行完整 Test**；应创建新的开发协议或只使用 Validation。完整运行只用于复现已冻结的 Text-only 与 Visual-only，不应加入新模型或据此调参。

机器可读正式报告位于 `outputs/final/frozen_final_test_v1.json`（本地生成、Git 忽略）。原冻结文件 SHA-256 为：

`f7e955c0fec0257ba7be67aab39b5d7531e587fe6234e149ab0d07275ecdcb85`

两次冻结实现的输出除运行时间外逐字段一致。

## 6. 结果证据

- 数据与时间边界：[M0_DATA_AUDIT.md](M0_DATA_AUDIT.md)
- 冻结切分：[split_v1.json](../configs/split_v1.json)
- 评测定义：[EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)
- Test 前冻结：[MODEL_FREEZE_V1.md](MODEL_FREEZE_V1.md)
- 最终结果：[FINAL_TEST_RESULTS.md](FINAL_TEST_RESULTS.md)

复现一致不意味着业务有效。这里的标签是公开评论正反馈，离线结果不能替代真实曝光日志和线上随机实验。

## 7. 本地只读 Demo

在项目根目录生成 8 个匿名 Validation 样例：

```powershell
.\.venv\Scripts\python.exe scripts\export_demo_data.py
```

启动本地页面：

```powershell
Set-Location demo
npm.cmd ci
npm.cmd run dev
```

访问 `http://localhost:3000/`。Node.js 要求 22.13 或更高版本。`demo/public/demo-data.json` 只在本地生成并被 Git 忽略，不包含原始用户 ID；完整选择规则见 [DEMO_PROTOCOL.md](DEMO_PROTOCOL.md)。
