# 项目验收证据索引

当前任务书 llm-lab/5.0：自动评分 95 分，人工仅验收“工程与优化解释”5 分（瓶颈与权衡 2、可追溯复现 2、下一步方案 1）。KV、prefix、attention 各 10 分由平台在评测时运行并从原始数据自动判定，并与最终服务配置联动；服务性能另占 30 分，使用固定负载和标定后的数字门槛。多用户凭据、计费与配额另占 15 分，包含 15 项 CPU 检查和对应真实部署门槛，见 [自动评分合同](autograding.md)。当前 7B/V5 标定状态见 [Daytona 记录](daytona-v5.md)。下表记录历史 A10 实测证据，不是当前 Daytona 已通过的声明，不另设人工分。本仓库没有上传或覆盖平台成绩的接口。历史 2.0 报告保留原来的 50/50 权重，不能视为新规则的自动 95 分。

| 验收范围 | 实现入口 | 实测与说明 | 状态 |
|---|---|---|---|
| 从空环境重建、幂等、profile 切换 | setup.sh、Dockerfile、supervisor.py | results/formal/report.json；实际冷启动 197.19 s | 正式自动评测通过 |
| 鉴权、模型信息、普通/流式 API | server/app.py、auth.py、backend.py | results/formal/declared-info.json；576 条原始请求 | 正式自动评测通过 |
| 有界队列、超时、断连、槽位释放 | admission.py、streaming.py、app.py | CPU tests；results/cancellation/ | 20 次真实取消通过 |
| Agent 真实模型与工具链 | agent/agent.py、agent/tools.py | results/formal/agent.json；源码和网关 model_call 计数 | 10 个随机任务通过；源码供复核 |
| KV 实现：同模型开关、eval、inference mode、上下文预算、释放路径 | experiments/kv_cache.py、configs/、server/ | manifest；cancellation；docs/architecture.md | 可执行；默认 BF16 检查如实失败 |
| KV 对照：128/512、固定32步、3重复、交替同步、内存 | experiments/kv_cache.py | results/kv/rehearsal/raw.jsonl | 全12条保留；不以耗时掩盖失败 |
| KV 机制：logits、容量、活跃/池/总显存、块分配与引用计数 | SOURCES.md；docs/performance.md | summary.json、manifest、取消后 engine 指标 | BF16 数值未通过；独立 FP32 控制通过，原始证据见 results/kv-diagnostics/ |
| 前缀部署：锁版本、on/off、清缓存、真实命令与计数 | experiments/prefix_cache.py | on/off-command.json、flush.json、metrics.txt | 通过 |
| 前缀对照：512前缀、8请求、shared/unique、3重复、cold/hot | 同上 | results/prefix/rehearsal/inputs.json、96条raw | 完成；聊天公共前缀没有隐藏 |
| 前缀机制：输出正确、token/hash、命中记录、源码、收益/占用 | SOURCES.md；docs/performance.md | cached_tokens；summary.json；engine metrics | 命中成立；小负载收益很小 |
| Attention 实现：强制math/flash、固定输入/掩码、软件硬件、服务后端 | experiments/flash_attention.py；configs/ | manifest、on/off命令、实际 Triton 服务配置 | 通过；不冒称正式服务使用SDPA Flash |
| Attention 对照：3shape、5预热、3×10、同步、allocated/reserved、失败处理 | 同上 | results/attention/rehearsal/raw.jsonl | 18条记录全部保留 |
| Attention 机制：shape、预设误差、seed/hash、trace、N²说明 | 同上；docs/performance.md | operators.json、6份trace.gz及SHA256 | 全部数值通过，后端证据齐全 |
| 工程解释：调用链、收益与代价、负结果、复现、来源 | docs/、SOURCES.md、README.md | 原始数据、图表、数据校验脚本 | 可复查；未完成项明确说明 |

可视化与证据核验在 CPU 上运行，不申请 GPU：

```bash
python3 -m experiments.verify_evidence
python3 -m experiments.overview
```

历史正式评分、优化实验、FP32 控制分别注明源代码 SHA。后续文档发布不是另一次 GPU 测量。上述 A10 记录尚未独立核对 usage；新的 Daytona v5 测试代码采用固定 tokenizer 独立计数，但结果需以新的运行证据为准。浏览器按钮链和更大模型/更长上下文的泛化效果不能由这些历史记录证明。
