# 自动评分与接口合同

## 自动优化评测接口（评分版本 llm-lab/4.0）

**自动 95 分 + 人工 5 分。** 人工仅验收“工程与优化解释”：瓶颈与权衡 2、可追溯复现说明 2、下一步方案 1。源码定位、取消机制和部署取舍写入这份说明，不另外设人工分。

点击“评测”后，平台在同一个全新 Daytona RTX 4090 沙箱上先运行服务/Agent 测试，再依次执行 KV、prefix、attention 三项独立实验；不会因 Git push、保存仓库或打开报告触发。完整评分仍受 30 分钟预算限制，实验超时/缺失会保留已完成项并明确扣分，不自动加时。KV/attention 各至多 300 秒，prefix 至多 480 秒，并受剩余总预算约束。

平台实际调用：

```bash
bash experiments/run.sh --case kv --out results/platform-<随机运行标识>/kv
bash experiments/run.sh --case prefix --out results/platform-<随机运行标识>/prefix
bash experiments/run.sh --case attention --out results/platform-<随机运行标识>/attention
```

入口必须支持任意上述新目录，不读取已有提交结果冒充新实测。可以停止自己的服务、运行实验；三个入口连续执行时必须均可工作。环境提供 MODEL_ID、MODEL_REVISION；固定模型为 Qwen/Qwen2.5-1.5B-Instruct，revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306。平台 clone 整个指定 commit，实验运行在 GPU，评分在控制端完成。实验退出非零时仍采集有效原始记录，所以数值错误不抹去其他通过项。

三项分别自动计 10 分，总计 30 分；最终服务性能另计 30 分，不再采用旧版“对照数据齐全即可得分”的规则。每项采用以下 6 个自动检查组，分值依次为 2/2/1/1/2/2。每组全部满足得满分，否则该组 0 分；报告逐组显示依据。原理实验没有加速不扣分，不因失败放宽容差；这不豁免最终服务的版本化吞吐/延迟门槛。原始数据是实验程序产生的测量证据，平台独立核对字段和计算，不将自报 points、quality_pass、summary.pass 或退出码直接当成成绩，也不宣称这些记录完全防伪。

参考实现的 KV 数学等价实验默认 FP32、禁用 TF32；更严格的 FP32 误差阈值保持不变。最终服务固定 BF16，服务正确性和性能由另一路独立测试。历史 BF16 微实验失败保留，显式 `KV_DTYPE=bfloat16` 可继续诊断，不将 FP32 结果冒称为 BF16 数值通过。

### KV Cache 与显存管理：10 分

| 自动检查 | 分值 | 判定条件 |
| --- | ---: | --- |
| 执行与溯源 | 2 | 本次独立 GPU 上执行；新目录、时间、commit、模型 revision、硬件、软件版本与完成记录匹配。数值失败导致非零退出不抹去已完成实验 |
| 实验矩阵覆盖 | 2 | 缓存开/关 × 长度 128/512 × 3 轮，共 12 条；每条固定 32 个 decode token；至少 1 次预热 |
| 输入一致性 | 1 | 每个长度的缓存开/关及各轮使用相同 prompt、固定 continuation 和 dtype |
| 性能原始数据 | 1 | 12 条原始耗时与 allocated/reserved 峰值均有效；不以加速倍数大小计分 |
| 输出与数值正确性 | 2 | 逐位置 logits 最大绝对误差：BF16 ≤ 0.15，FP32 ≤ 0.001；top-1 一致率 100%。固定容差，不因失败放宽 |
| 机制证据 | 2 | 独立核对指定模型 2×L×Hkv×dh×字节数×token 数的 KV 容量，含 1/8 请求；allocator reserved 不当作有效 KV；同时通过下方最终部署 KV 配置门槛 |

### RadixAttention 与前缀复用：10 分

| 自动检查 | 分值 | 判定条件 |
| --- | ---: | --- |
| 执行与溯源 | 2 | 本次独立 GPU 上执行；新目录、时间、commit、模型 revision、硬件、软件版本与完成记录匹配。数值失败导致非零退出不抹去已完成实验 |
| 实验矩阵覆盖 | 2 | 开/关 × 共享/不同前缀 × 3 轮 × 8 请求，共 96 条；每轮清缓存，首条冷缓存 |
| 输入一致性 | 1 | 保存完整 token IDs；独立重算哈希；共享组恰好 512 前缀 token，不同组八个前缀互异，全文等长 |
| 性能原始数据 | 1 | 96 条 TTFT/E2E 和 cached_tokens 有效，12 次清缓存成功记录完整；goodput 可为零 |
| 输出与数值正确性 | 2 | 直接核对全部 96 个原始输出为 OK，传输成功；不采信自报 quality_pass |
| 机制证据 | 2 | 关缓存命中为 0；共享热请求复用 ≥512 tokens；每轮首条冷请求及不同前缀均 <512；不要求必然加速；同时通过下方最终部署 prefix 配置门槛 |

### FlashAttention 与算子优化：10 分

| 自动检查 | 分值 | 判定条件 |
| --- | ---: | --- |
| 执行与溯源 | 2 | 本次独立 GPU 上执行；新目录、时间、commit、模型 revision、硬件、软件版本与完成记录匹配。数值失败导致非零退出不抹去已完成实验 |
| 实验矩阵覆盖 | 2 | math/flash × N=128/512/1024 × 3 轮，共 18 条；预热 5 次，每轮 10 次；BF16、causal、dropout=0 |
| 输入一致性 | 1 | B=1,H=8,dh=64；同一长度的 Q/K/V 哈希、输出形状在全部轮次和后端匹配 |
| 性能原始数据 | 1 | 18 条计时与显存峰值有效；同步后计时，性能提升幅度不计分 |
| 输出与数值正确性 | 2 | math 与 flash 输出最大绝对误差 ≤ 0.02，固定容差 |
| 机制证据 | 2 | 三个长度的 profiler 算子记录分别含实际 math 与 FlashAttention kernel；这是微实验，不冒充线上服务后端；同时通过下方最终部署 attention 配置门槛 |

### 原理实验与最终部署联合验收

每项的“机制证据”2 分必须同时满足原始实验检查和最终服务配置检查：

| 项目 | optimized 服务配置要求 |
| --- | --- |
| KV Cache | SGLang 的 `max_total_tokens` 是正整数，且大于 baseline 的正整数预算 |
| RadixAttention | `prefix_cache=true` |
| Attention | 本期 RTX 4090 使用 `attention_backend=flashinfer` |

`/info` 必须报告实际 SGLang 版本、optimized profile、指定模型与 revision。FlashInfer 是此最终 SGLang 服务使用的高效 Attention 后端；独立 PyTorch FlashAttention 实验仍用于数学、数值和算子比较，不把两者混称为同一个实现。

配置检查来源为 `service_declared_runtime_configuration`，属于服务自报的运行配置辅助证据，不是完整防伪证明。平台保存配置检查、独立性能实测和原理实验记录。缺少最终配置时，这 2 分不得分；配置声明通过也不能替代原理证据或数字性能达标。不要求每项独立优化都必然提速，需解释收益与限制。

平台由 `grader.py::deployment_checks` 保存 `deployment-checks.json`，并写入 `report.deployment_checks`；`gate_deployed_mechanism` 将每项机制 2 分与该配置结果联动。原始实验通过但配置不通过，机制项仍为 0；其他已通过检查保留。报告逐项保存 `deployment_passed`，UI 同时展示最终配置表与实验细项。

### 证据文件格式

每个 `--out` 目录必须生成 UTF-8 JSON 文件，不允许符号链接，每个被采集文件最多 256,000 字节；未完成字段写 null，禁止 NaN/Infinity。所有原始行的分组键必须唯一，缺失/重复/额外样本都会使矩阵检查失败。

`manifest.json`：`case`（kv/prefix/attention）、`time`（本次 UTC ISO 时间）、`commit`（完整 SHA）、`model`、`revision`、`gpu`（`nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader` 原始文本去尾部换行）、`software`（torch/transformers/sglang 版本）、`cuda`、`complete`（只有完整运行才为 true）、`parameters`。数值不通过仍应保留 complete=true，并返回非零；中途失败为 false。

- KV parameters：lengths=[128,512]、decode_tokens=32、warmup≥1、repeats=3、dtype=bfloat16 或 float32。raw.jsonl 共 12 行：variant=cache_on/cache_off、length、repeat=0/1/2、input_hash、continuation_hash、decode_tokens、dtype、elapsed_ms、peak_allocated_bytes、peak_reserved_bytes、max_abs_error、atol、rtol、top1_agreement。同一长度的 prompt/continuation 哈希必须一致。summary.json：layers=28、KV_heads=2、head_dim=128、dtype_bytes（2/4）、KV_bytes_per_token；measurements 两项包含 length、KV_bytes_one_request、KV_bytes_eight_requests（包含额外 32 token）。逐位置比较 logits，记录全部位置最大误差；评分采用保守的最大绝对误差阈值，而不是采信自报 allclose。
- Attention parameters：warmup≥5、groups=3、iterations=10、dtype=bfloat16、causal=true、dropout=0。raw.jsonl 共 18 行：variant=math/flash、length、repeat、shape、output_shape、input_hashes（Q/K/V 三个 SHA256）、elapsed_ms、peak_allocated_bytes、peak_reserved_bytes、max_abs_error、atol=.02、rtol=.02。另存 math-128-operators.json、flash-128-operators.json 等三个长度共 6 个文件，内容是 profiler 算子名称字符串数组；保存完整 trace 供追溯。
- Prefix parameters：prefix_tokens=512、requests=8、repeats=3。raw.jsonl 共 96 行：variant=on/off、group=shared/unique、repeat=0/1/2、index=0…7、input_hash、prefix_hash、prefix_length=512、input_tokens、cached_tokens、cache_state、output、transport_ok、ttft_ms、e2e_ms。每轮 index=0 为 cold，开启缓存且共享前缀 index>0 为 hot。inputs.json 为 `{ "shared": [8个token ID数组], "unique": [8个token ID数组] }`；hash 使用 Python `sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()`。每个 on/off、shared/unique、repeat 组合保存 `<variant>-<group>-<repeat>-flush.json`，至少包含 status=200。实际引擎命中计数来自服务结果，不能由速度推断。summary.json 保留各组窗口、成功率、质量 goodput；不得隐去零 goodput。

张量哈希使用连续张量原始字节的 SHA256；Prefix 使用上面的 JSON 序列化。保存原始失败记录及图表，docs/performance.md 链接同一 commit 的证据。未执行或格式错误的组不得显示“通过”。

## 最终服务性能与工程行为

最终服务由控制端 `serving_benchmark.py` 和 `serving_probes.py` 独立测试。版本化 `acceptance.json` 固定硬件、模型/revision、BF16、工作负载和并发；当前标定未完成时 targets=null，性能项为 0 且 status=uncalibrated，不能生成通过结论。标定完成后才能发布数字门槛；不复用历史 Lambda A10 数据。

性能 30 分按配置中的目标组等分。每组必须同时通过输出 token/s、TTFT p95、TPOT p95、E2E p95、goodput、SLO 达标比例、成功率与正确率；还需匹配硬件/模型/精度、足够采样、完整时序和控制端 tokenizer 计数。失败组不能用更快的其他指标抵消。输出 token/s 只统计正确成功输出；SSE 块间隔不宣称精确 token ITL。

API 10 分为鉴权 2、模型信息 1、非流式 1、SSE 2、过载拒绝 2、取消与恢复 2。工程探针在性能窗口之外执行。服务声明的 active/queued/abort_acknowledged 仅辅助核对；平台实际发起突发请求、关闭流、观察队列排空并检查恢复请求。人工分析仍只有 5 分，不用人工替代未通过的自动测试。

历史 llm-lab/3.0 的每项 15 分原理实验、llm-lab/2.0 的 50/50 与 llm-lab/1.0 的 80/20 保留在原报告中，不按本版本追溯改分。
