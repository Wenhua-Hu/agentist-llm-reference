

## 自动优化评测接口（评分版本 llm-lab/3.0）

**自动 95 分 + 人工 5 分。** 人工仅验收“工程与优化解释”：瓶颈与权衡 2、可追溯复现说明 2、下一步方案 1。源码定位、取消机制和部署取舍写入这份说明，不另外设人工分。

点击“评测”后，平台在同一台全新 GPU 上先运行服务/Agent 测试，再依次执行 KV、prefix、attention 三项独立实验；不会因 Git push、保存仓库或打开报告触发。完整评分仍受 30 分钟预算限制，实验超时/缺失会保留已完成项并明确扣分，不自动加时。KV/attention 各至多 300 秒，prefix 至多 480 秒，并受剩余总预算约束。

平台实际调用：

```bash
bash experiments/run.sh --case kv --out results/platform-<随机运行标识>/kv
bash experiments/run.sh --case prefix --out results/platform-<随机运行标识>/prefix
bash experiments/run.sh --case attention --out results/platform-<随机运行标识>/attention
```

入口必须支持任意上述新目录，不读取已有提交结果冒充新实测。可以停止自己的服务、运行实验；三个入口连续执行时必须均可工作。环境提供 MODEL_ID、MODEL_REVISION；固定模型为 Qwen/Qwen2.5-1.5B-Instruct，revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306。平台 clone 整个指定 commit，实验运行在 GPU，评分在控制端完成。实验退出非零时仍采集有效原始记录，所以数值错误不抹去其他通过项。

每项采用以下 6 个自动检查组。每组全部满足得满分，否则该组 0 分；报告逐组显示依据。结果没有加速不扣分，不因失败放宽容差。原始数据是实验程序产生的测量证据，平台独立核对字段和计算，不将自报 points、quality_pass、summary.pass 或退出码直接当成成绩，也不宣称这些记录完全防伪。

### KV Cache 与显存管理：15 分

| 自动检查 | 分值 | 判定条件 |
| --- | ---: | --- |
| 执行与溯源 | 3 | 本次独立 GPU 上执行；新目录、时间、commit、模型 revision、硬件、软件版本与完成记录匹配。数值失败导致非零退出不抹去已完成实验 |
| 实验矩阵覆盖 | 4 | 缓存开/关 × 长度 128/512 × 3 轮，共 12 条；每条固定 32 个 decode token；至少 1 次预热 |
| 输入一致性 | 2 | 每个长度的缓存开/关及各轮使用相同 prompt、固定 continuation 和 dtype |
| 性能原始数据 | 2 | 12 条原始耗时与 allocated/reserved 峰值均有效；不以加速倍数大小计分 |
| 输出与数值正确性 | 2 | 逐位置 logits 最大绝对误差：BF16 ≤ 0.15，FP32 ≤ 0.001；top-1 一致率 100%。固定容差，不因失败放宽 |
| 机制证据 | 2 | 独立核对指定模型 2×L×Hkv×dh×字节数×token 数的 KV 容量，含 1/8 请求；allocator reserved 不当作有效 KV |

### RadixAttention 与前缀复用：15 分

| 自动检查 | 分值 | 判定条件 |
| --- | ---: | --- |
| 执行与溯源 | 3 | 本次独立 GPU 上执行；新目录、时间、commit、模型 revision、硬件、软件版本与完成记录匹配。数值失败导致非零退出不抹去已完成实验 |
| 实验矩阵覆盖 | 4 | 开/关 × 共享/不同前缀 × 3 轮 × 8 请求，共 96 条；每轮清缓存，首条冷缓存 |
| 输入一致性 | 2 | 保存完整 token IDs；独立重算哈希；共享组恰好 512 前缀 token，不同组八个前缀互异，全文等长 |
| 性能原始数据 | 2 | 96 条 TTFT/E2E 和 cached_tokens 有效，12 次清缓存成功记录完整；goodput 可为零 |
| 输出与数值正确性 | 2 | 直接核对全部 96 个原始输出为 OK，传输成功；不采信自报 quality_pass |
| 机制证据 | 2 | 关缓存命中为 0；共享热请求复用 ≥512 tokens；每轮首条冷请求及不同前缀均 <512；不要求必然加速 |

### FlashAttention 与算子优化：15 分

| 自动检查 | 分值 | 判定条件 |
| --- | ---: | --- |
| 执行与溯源 | 3 | 本次独立 GPU 上执行；新目录、时间、commit、模型 revision、硬件、软件版本与完成记录匹配。数值失败导致非零退出不抹去已完成实验 |
| 实验矩阵覆盖 | 4 | math/flash × N=128/512/1024 × 3 轮，共 18 条；预热 5 次，每轮 10 次；BF16、causal、dropout=0 |
| 输入一致性 | 2 | B=1,H=8,dh=64；同一长度的 Q/K/V 哈希、输出形状在全部轮次和后端匹配 |
| 性能原始数据 | 2 | 18 条计时与显存峰值有效；同步后计时，性能提升幅度不计分 |
| 输出与数值正确性 | 2 | math 与 flash 输出最大绝对误差 ≤ 0.02，固定容差 |
| 机制证据 | 2 | 三个长度的 profiler 算子记录分别含实际 math 与 FlashAttention kernel；这是微实验，不冒充线上服务后端 |

### 证据文件格式

每个 `--out` 目录必须生成 UTF-8 JSON 文件，不允许符号链接，每个被采集文件最多 60,000 字节；未完成字段写 null，禁止 NaN/Infinity。所有原始行的分组键必须唯一，缺失/重复/额外样本都会使矩阵检查失败。

`manifest.json`：`case`（kv/prefix/attention）、`time`（本次 UTC ISO 时间）、`commit`（完整 SHA）、`model`、`revision`、`gpu`（`nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader` 原始文本去尾部换行）、`software`（torch/transformers/sglang 版本）、`cuda`、`complete`（只有完整运行才为 true）、`parameters`。数值不通过仍应保留 complete=true，并返回非零；中途失败为 false。

- KV parameters：lengths=[128,512]、decode_tokens=32、warmup≥1、repeats=3、dtype=bfloat16 或 float32。raw.jsonl 共 12 行：variant=cache_on/cache_off、length、repeat=0/1/2、input_hash、continuation_hash、decode_tokens、dtype、elapsed_ms、peak_allocated_bytes、peak_reserved_bytes、max_abs_error、atol、rtol、top1_agreement。同一长度的 prompt/continuation 哈希必须一致。summary.json：layers=28、KV_heads=2、head_dim=128、dtype_bytes（2/4）、KV_bytes_per_token；measurements 两项包含 length、KV_bytes_one_request、KV_bytes_eight_requests（包含额外 32 token）。逐位置比较 logits，记录全部位置最大误差；评分采用保守的最大绝对误差阈值，而不是采信自报 allclose。
- Attention parameters：warmup≥5、groups=3、iterations=10、dtype=bfloat16、causal=true、dropout=0。raw.jsonl 共 18 行：variant=math/flash、length、repeat、shape、output_shape、input_hashes（Q/K/V 三个 SHA256）、elapsed_ms、peak_allocated_bytes、peak_reserved_bytes、max_abs_error、atol=.02、rtol=.02。另存 math-128-operators.json、flash-128-operators.json 等三个长度共 6 个文件，内容是 profiler 算子名称字符串数组；保存完整 trace 供追溯。
- Prefix parameters：prefix_tokens=512、requests=8、repeats=3。raw.jsonl 共 96 行：variant=on/off、group=shared/unique、repeat=0/1/2、index=0…7、input_hash、prefix_hash、prefix_length=512、input_tokens、cached_tokens、cache_state、output、transport_ok、ttft_ms、e2e_ms。每轮 index=0 为 cold，开启缓存且共享前缀 index>0 为 hot。inputs.json 为 `{ "shared": [8个token ID数组], "unique": [8个token ID数组] }`；hash 使用 Python `sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()`。每个 on/off、shared/unique、repeat 组合保存 `<variant>-<group>-<repeat>-flush.json`，至少包含 status=200。实际引擎命中计数来自服务结果，不能由速度推断。summary.json 保留各组窗口、成功率、质量 goodput；不得隐去零 goodput。

张量哈希使用连续张量原始字节的 SHA256；Prefix 使用上面的 JSON 序列化。保存原始失败记录及图表，docs/performance.md 链接同一 commit 的证据。未执行或格式错误的组不得显示“通过”。
