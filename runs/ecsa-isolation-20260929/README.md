# ECSA 补充消融实验（2026-09-29）

本目录保存新增实验，不覆盖原论文、表 4 或 `runs/dbmags-ablation/full_report_llm_xhigh.json`。

**144/144 次预定调用已完成。先阅读 [实验结果](实验结果.md)，受控比较以 `controlled_analysis.json` 为准。**

## 历史复用与受控变化

历史启动记录位于 2026-08-12 长会话的第 25998 行（9 月 v11 重跑阶段）：使用 `rca39`、冻结 v11 数据、seed 20260802、GPT-5.5 / xhigh。原六折每折训练 550、测试 110，共 660 例；每折最多 12 次测试仲裁。

本次直接复用相同的数据加载器、外层划分、超图训练、EPDG、本地判别规则及证据卡片构造。重新拟合后，660 条超图预测与历史报告逐条一致。

1. **本地判别单独输出**：沿用既有规则；若该规则提名了具有直接证据或指标模式优势的 challenger，就选择它；否则保留原超图第一候选。这是明确的确定性输出适配，不再采用历史“仅生成候选、最终输出仍保持不变”的读出方式，也未另训分类器或调整阈值。
2. **仅 LLM**：输入完整的四候选池，保留统一构造的支持/反证卡片。
3. **完整筛选＋LLM**：输入本地保留的原第一候选及至多一个 challenger。
4. 两条 LLM 路径采用同一提示词、结构化输出契约、同一证据构造和相同的 72 个样本（每折 12 个），共 144 次逻辑调用。样本按超图前两候选的概率差从小到大选取，不使用测试标签或本地判别结果选择样本。两组均关闭原先不同的按折校准开关。
5. 仲裁弃权、非法输出、调用失败均保留超图原第一候选。未安排仲裁的样本也保留这一共同基线。每个样本与方法仅保留一次预定调用结果，不因结果不好而重试。

## 官方登录接入

用户明确要求改用当前官方会话的登录通道。使用已登录 ChatGPT 的本机 Codex CLI 进行独立、临时、禁止工具使用的模型推断；保留 GPT-5.5 / xhigh。工作目录为空，仅传入盲化后的候选和证据，不传入本研究对话、测试标签、场景、原始 SQL 或案例 ID。没有提取登录令牌，也没有将其交给第三方接口。

由于本机 WebSocket 握手超时，使用官方默认地址与登录处理的 HTTPS 通道。CLI 的系统包装与历史原始 Responses API 请求不同，因此新旧结果分别保留，组件效果主要比较本次同条件的两组。

官方功能说明：[非交互调用](https://learn.chatgpt.com/docs/non-interactive-mode)、[登录方式](https://learn.chatgpt.com/docs/auth)。实际模型与接入参数以 `llm_configuration.json` 为准。

## 如何看结果

- `deterministic_report.json`：全部 660 例本地消融，以及候选数量、真值保留率。
- `prepared.json`：冻结的全部预测输入、共同样本安排、候选池及盲化请求；真值仅在本地评分记录中，API 仅收到各 job 的 `packet`。
- `llm_decisions.jsonl`：逐次调用的有效结构化决策、状态与 token 用量；可断点继续，不保存模型原始推理。
- `full_report.json`：全部调用完成后产生。
- `controlled_analysis.json`：受控比较的最终分析。优先查看相同 72 例上的四组；同时给出全部 660 例中仅允许同样 72 例改变预测的结果。本地规则全量运行的 81.67% 属于另一覆盖范围，不能直接与限额调用的整体 LLM 成绩比较来判断模型优劣。
- `independent_count_check.json`：独立计算 TP/FP/FN、Exact 和 micro-F1 的核验。
- `independent_final_count_check.json`：全部调用完成后的独立计数与三种评估范围核验。
- `paired_case_audit.csv`：共同 72 例的真值、四组输出、候选保留状态及调用状态。
- `packet_pair_audit.json`：全部实际请求的成对证据一致性与盲化字段核验。
- `protocol.json`：固定协议、数据/代码/历史报告/论文指纹。`protocol_before_official_transport.json` 保留用户指定官方通道前的准备记录；候选和样本安排未因接入方式而改变。

这次 LLM 对照覆盖 72 个按上游不确定性选择的样本，不是全 660 例 LLM 推断。六折仍包含相同 22 个场景，不能解释为未见场景泛化。每个样本每组一次模型推断；报告的置信区间以六个采集重复折为分块，统计解释仍受折数限制。

## 复现与续跑

首次准备（已完成，勿覆盖；新实验用另一个 `--output`）：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ./rca39/bin/python \
  src/ecsa_isolation_experiment.py prepare --seed 20260802 --budget 12
```

官方登录通道运行/续跑：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ./rca39/bin/python \
  src/ecsa_isolation_experiment.py run --transport codex
```

全部调用完成后分析：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ./rca39/bin/python \
  src/analyze_ecsa_isolation.py
```

已通过新增协议校验、本地判别器和原仲裁器相关的 16 项测试。全部预定调用、独立指标核验及数据／代码／论文指纹核验均已完成。
