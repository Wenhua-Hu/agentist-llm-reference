# 自动评分与接口合同


## 当前评分总表与验证状态

当前公开参考分支为 [codex/credential-service-v5](https://github.com/mrvgao/agentist-llm-reference/tree/codex/credential-service-v5)，默认分支仍保留旧实现。部署时按平台任务书给出的完整 commit 固定源码；本文件不更新该冻结提交。本轮已完成独立确认，自动 95/95、人工 5 分待验收；实测源码与后续文档提交分开标记，见 [Daytona V5 实测记录](daytona-v5.md)。

| 项目 | 分值 | 自动验收内容 |
| --- | ---: | --- |
| 干净重建 | 5 | 首次启动 2.5、幂等重跑 2.5 |
| API 与工程行为 | 5 | 鉴权 1、模型信息 0.5、非流式 0.5、SSE 1、过载 1、取消与恢复 1 |
| 服务正确性 | 5 | optimized 全部请求成功率 ≥95% 且正确率 ≥90%，同时满足得 5 分，否则 0 |
| 最终服务性能 | 30 | 四组吞吐/延迟/质量联合目标；未标定或未确认时待评定 |
| Agent 接入 | 5 | 模型规划、工具执行与最终回答走自己的服务 |
| KV Cache | 10 | 独立原始实验与最终部署机制证据 |
| RadixAttention | 10 | 独立原始实验与最终部署机制证据 |
| FlashAttention | 10 | 独立原始实验与最终部署机制证据 |
| 多用户凭据、计费与配额 | 15 | 三组各 5 分，15 项 CPU 检查与对应真实 SGLang 部署门槛联合验收 |
| 工程与优化解释 | 5（人工） | 瓶颈/权衡 2、复现说明 2、下一步方案 1 |

前九项自动合计 95 分，第十项人工仅 5 分。性能 `uncalibrated` / `unconfirmed` 表示平台门槛尚未就绪，页面标为“待评定”，不能显示为学员失败 0/30，也不生成正式最终总分；其他已经执行的成绩仍保留。控制器中断或失租同样不生成最终成绩，原始数据仅用于排查。提交未实现接口、setup 失败与业务断言失败仍按未完成交付计分，不能当作平台待标定而免扣分。

## 自动优化评测接口（评分版本 llm-lab/5.0）

**自动 95 分 + 人工 5 分。** 人工仅验收“工程与优化解释”：瓶颈与权衡 2、可追溯复现说明 2、下一步方案 1。源码定位、取消机制和部署取舍写入这份说明，不另外设人工分。

点击“评测”后，平台在同一个全新 Daytona RTX 4090 沙箱上先运行服务/Agent 测试，再依次执行 KV、prefix、attention 三项独立实验；不会因 Git push、保存仓库或打开报告触发。完整评分受 60 分钟预算限制，实验超时/缺失会保留已完成项并明确扣分，不自动加时。KV/attention 各至多 300 秒，prefix 至多 480 秒，并受剩余总预算约束。

平台实际调用：

```bash
bash experiments/run.sh --case kv --out results/platform-<随机运行标识>/kv
bash experiments/run.sh --case prefix --out results/platform-<随机运行标识>/prefix
bash experiments/run.sh --case attention --out results/platform-<随机运行标识>/attention
```

入口必须支持任意上述新目录，不读取已有提交结果冒充新实测。可以停止自己的服务、运行实验；三个入口连续执行时必须均可工作。平台按 case 提供 MODEL_ID、MODEL_REVISION：kv 独立固定 Qwen/Qwen2.5-1.5B-Instruct，revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306，FP32；prefix 使用候选服务 Qwen/Qwen2.5-7B-Instruct，revision a09a35458c702b33eeacc393d103063234e8bc28，BF16；attention 为合成张量实验，以同一 7B 身份标记服务上下文，不加载权重。入口拒绝不匹配的显式模型/revision，不能把 7B 的服务参数传给 kv。平台 clone 整个指定 commit，实验运行在 GPU，评分在控制端完成。实验退出非零时仍采集有效原始记录，所以数值错误不抹去其他通过项。

三项分别自动计 10 分，总计 30 分；最终服务性能另计 30 分，不再采用旧版“对照数据齐全即可得分”的规则。每项采用以下 6 个自动检查组，分值依次为 2/2/1/1/2/2。每组全部满足得满分，否则该组 0 分；报告逐组显示依据。原理实验没有加速不扣分，不因失败放宽容差；这不豁免最终服务的版本化吞吐/延迟门槛。原始数据是实验程序产生的测量证据，平台独立核对字段和计算，不将自报 points、quality_pass、summary.pass 或退出码直接当成成绩，也不宣称这些记录完全防伪。

参考实现的 KV 数学等价实验默认 FP32、禁用 TF32；更严格的 FP32 误差阈值保持不变。最终服务固定 BF16，服务正确性和性能由另一路独立测试。历史 BF16 微实验失败保留；当前 KV 入口只接受 FP32，不将 FP32 结果冒称为 BF16 数值通过。

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

最终服务由控制端 `serving_benchmark.py` 和 `serving_probes.py` 独立测试。版本化 `acceptance.json` 固定硬件、模型/revision、BF16、工作负载和并发；当前标定未完成时 targets=null、status=uncalibrated，性能 30 分待评定，不生成正式最终总分，也不能生成通过结论。候选门槛尚未独立确认（unconfirmed）时采用相同语义。标定完成后才能发布数字门槛；不复用历史 Lambda A10 数据。

性能 30 分固定为 short/long × 并发 8/16 四组，每组 7.5 分，不允许删掉较难的一组再重分权重。每组必须同时通过输出 token/s、TTFT p95、TPOT p95、E2E p95、goodput、SLO 达标比例、成功率与正确率；还需匹配硬件/模型/精度、足够采样、完整时序和控制端 tokenizer 计数。失败组不能用更快的其他指标抵消。输出 token/s 只统计正确成功输出；SSE 块间隔不宣称精确 token ITL。

API 5 分为鉴权 1、模型信息 0.5、非流式 0.5、SSE 1、过载拒绝 1、取消与恢复 1。工程探针在性能窗口之外执行。服务声明的 active/queued/abort_acknowledged 仅辅助核对；平台实际发起突发请求、关闭流、观察队列排空并检查恢复请求。人工分析仍只有 5 分，不用人工替代未通过的自动测试。

历史 llm-lab/4.0 的重建/API/正确性各 10 分且未覆盖管理服务、llm-lab/3.0 的每项 15 分原理实验、llm-lab/2.0 的 50/50 与 llm-lab/1.0 的 80/20 保留在原报告中，不按本版本追溯改分。

### V5 负载与流式测量边界

固定服务为 Qwen2.5-7B-Instruct BF16、revision `a09a35458c702b33eeacc393d103063234e8bc28`、RTX 4090。负载 `typed-json-numbered-records-prose-exact-input/5` 使用同一自然语言前缀库与 JSON 模式：short 是 512 个 chat-template 输入 tokens、1 条记录、最多输出 96 tokens；long 是 1024 输入、8 条带编号记录、最多输出 384。编号只表达输入顺序，不能加入答案；严格校验 JSON 类型、字段、值及数组顺序，不移除围栏、不修复数字、不忽略排序错误。JSON 模式不能保证语义正确。

每个 profile 的每种长度在并发 1/4/8/16 各测 3 轮，每轮 16 个请求，预热另记。两 profile 使用相同测量输入，closed-loop worker 完成一个请求再发下一个；共享前缀允许缓存命中，因此结果不代表通用聊天吞吐。校准/独立确认要求 optimized 全 8 组以及 baseline 并发 1 的两组全部正确成功；baseline 高并发只允许可识别且经策略审核的容量失败，不能将答案错误、503 或通用协议失败视为正常超时。正式四组目标成功率与正确率各 ≥99%；每组 48 个样本意味着必须 48/48。

流式性能请求要求 HTTP 200、text/event-stream、非空文本、finish_reason 与 [DONE]，error 或缺结束标记视为失败。明确 request_timeout/upstream_timeout 仍是失败，计入所有请求分母；部分输出不计有效吞吐，旧的未知协议错误不能改分类。TTFT 从客户端发起请求到首个非空文本事件；E2E 到消费结束；TPOT 是首末内容事件间隔除以独立重分词可见输出 tokens 减一，只是代理值。只有一个缓冲内容事件时不伪造 TPOT=0。SSE 事件间隔不是逐 token ITL。计费 live gate 另要求完整 usage；不要混淆性能可见 token 统计与精确收费。

首轮候选及独立确认状态、测量 SHA 与文档发布 SHA 的边界见 [Daytona V5 实测记录](daytona-v5.md)。

## 管理服务 15 项协议与联合部署门槛

原 12 个需求场景见 [credential-service/1](credential-service.md)。以下与平台当前合同一致。


平台独立运行 `controller-mock/1`，向你的 HTTP 网关发起请求并核对结果，不读取你自报的测试分数。以下 **15 项各 1 分（CPU 原始检查）**，归入三组各 5 分，并须通过对应真实 GPU 部署检查才计入最终分；这些是验收要求，不是已经通过的实测记录。

#### 启动、停止与崩溃恢复入口

`setup.sh` 除正常 SGLang 启动外，必须支持 `bash setup.sh --contract-test`。此模式**只替换模型引擎**，实际用户鉴权、密钥摘要校验、租户隔离、限流、余额预占、用量账本、结算和持久化必须使用与正式服务相同的实现。不得识别评测请求返回特制答案、跳过计费，或把 contract-test 视为免鉴权后门。

平台先启动独立的确定性模型 fixture，再提供以下环境变量：

| 变量 / 文件 | 约定 |
| --- | --- |
| `ENGINE_BASE_URL` | `http://127.0.0.1:30001`；连接平台提供的模型 fixture，不自行启动或替换它 |
| `PORT` | 本轮网关监听 `8000`，实际由 `PORT` 读取 |
| `API_KEY` | 初始化普通 benchmark 用户的推理 key，仍走真实鉴权、额度与账本 |
| `ADMIN_API_KEY`、`KEY_HASH_SECRET` | 与正式模式相同的管理鉴权和摘要配置；从环境变量读取，不打印、持久化原文或写入仓库 |
| `DATA_DIR` | 平台指定的本轮测试数据目录；同目录重启/幂等 setup 保留账本、用户、key 摘要与价格版本 |
| `$DATA_DIR/gateway.pid` | 写入实际占用网关端口的进程 PID，不是启动 shell、fixture 或其他进程 PID；每次启动后更新 |

`bash setup.sh --stop` 只停止自己启动的网关/服务进程，不清空 `DATA_DIR`，不终止平台 fixture；禁止按端口或进程名批量杀掉其他进程。正式模式仍启动真实 SGLang，不把 fixture 当成 GPU 部署。

fixture 的 `POST /tokenize` 返回 `count=100`（并提供 token ID 数组）；推理返回受控的 `usage`：输入 100、输出 50。初始输入/输出价格分别为 **2 / 6 micro-USD/token**，一次费用为 `100×2+50×6=500 micro-USD`。fixture 可延迟响应、在 usage 前中断、或先发最终 usage 再延迟 `[DONE]`。这些确定性数据只用于事务验收，不用于性能分或真实模型正确性评分。 fixture 提供 `GET /health`、`GET /v1/models`、`POST /tokenize`、`POST /v1/chat/completions` 和 `POST /abort_request`（使用对应请求的 `rid`）。`/__fixture/*` 仅供控制端配置和观察，提交服务不能依赖这些管理接口或获取 fixture 管理凭据。

恢复测试先核对 `gateway.pid` 对应进程确实持有 8000 监听 socket，再对该网关注入 `SIGKILL`，使用相同环境变量和 `DATA_DIR` 重跑 `setup.sh --contract-test`。要求覆盖两个窗口：① 预占后尚无可信 usage，保留未知用量与调整审计，60 秒内释放预占；② 最终 usage 已到达网关、但流尚未 `[DONE]`，恢复后根据持久化 usage 恰好结算一次。已提交账目不能丢失或重复扣费。

#### 15 项检查与原 12 个需求场景的对应关系

| test_id | 自动组 / 分值 | 独立测试内容 | 对应需求场景 |
| --- | --- | --- | --- |
| `isolation` | key_lifecycle / 1 | 两租户各自请求与账目隔离；普通 key 不能读/写管理接口或覆盖 tenant_id；7 个管理写路由被拒且状态不变 | 1、2 |
| `storage` | key_lifecycle / 1 | key 列表不回传原文；扫描 DATA_DIR 未发现本次签发的原文 key | 2 |
| `rotate` | key_lifecycle / 1 | 新 key 与 key_id 不同；旧 key 新请求 401，新 key 成功 | 3 |
| `revoke_expiry` | key_lifecycle / 1 | 撤销与到期 key 均返回 401 | 4 |
| `disabled` | key_lifecycle / 1 | 禁用租户推理返回 403 | 4 |
| `nonstream_billing` | metering_billing / 1 | 非流式 100/50 tokens 精确结算 500；余额与预占、逐 key 明细一致 | 5 |
| `stream_billing` | metering_billing / 1 | 客户端未请求 usage 时，流式仍记录 100/50 并精确结算 500 | 5、9 |
| `idempotency` | metering_billing / 1 | 20 次同 Idempotency-Key 请求至多一次模型生成/扣费；不同请求体复用返回 409 | 6 |
| `pricing` | metering_billing / 1 | 价格改为 4/8 后新费用 800、历史仍 500，版本独立，累计 1300 | 11 |
| `credits` | metering_billing / 1 | 相同充值 key 仅入账一次；同 key 改金额返回 409 | 11 |
| `reservation` | quota_recovery / 1 | 余额只够一次时并发 20 请求，仅一次进入模型并结算，其余 402；预占归零 | 7 |
| `token_quota` | quota_recovery / 1 | 多 key 共享用户累计 token 额度，150 用完后另一 key 返回 402 | 8 |
| `rpm` | quota_recovery / 1 | 多 key 共享 RPM，超限 429 + Retry-After | 8 |
| `cancel` | quota_recovery / 1 | 首段断流后上游 abort、并发名额恢复；取消账目保留，未知 usage 不伪造 | 8、9 |
| `recovery` | quota_recovery / 1 | 两个真实网关崩溃窗口：无 usage 的调整审计，以及已 journal usage 的恰好一次结算 | 10 |

这张映射不意味着每项需求的所有安全属性都由黑盒证明。例如原场景 1 的“两用户各两把 key”、完整日志脱敏、轮换前历史记录可追溯、管理网页全部操作，仍须交付对应代码与自测证据；控制端的 `storage` 只扫描本次签发原文 key 是否出现在 DATA_DIR 和列表中，**不证明密码学摘要算法、随机性或所有日志都安全**。源码仍须实现带服务端 secret 的安全摘要，不能可逆保存原文 key。

原场景 12 的真实 GPU 流式、非流式、Agent 调用继续由 live 自测和实际服务评测验证，不能拿这 15 项 mock 测试代替。性能压测必须保持正式鉴权、配额与账本开启。

#### 与真实 SGLang 部署联合验收

15 项 CPU 确定性检查不是独立的模拟项目。每个 5 分组还必须通过对应的 **optimized 真实 GPU 服务部署检查**才保留该组 CPU 得分；对应部署检查失败或未完成，该组为 0，并标注“部署未验证”。原始 CPU case 结果仍保留用于定位问题，不把其通过状态改写成已获得最终组分。

真实服务检查会创建两个独立用户，分别完成流式和非流式调用，核对首个内容事件、finish_reason、[DONE]、报告 usage 与费用/用户账本及互不可见；测试轮换后旧 key 401、新 key 可调用；并确认原 `API_KEY` benchmark 用户的 `/v1/usage` 中已有真实模型调用账目。正式性能、正确性与 Agent 测试不能绕开鉴权、配额和账本。

| CPU 组 | 对应真实 optimized 网关联合门槛 |
| --- | --- |
| key_lifecycle | 两租户各流式/非流式、账目归属与互不可见、普通 key 无管理权限，7 个管理写路由返回 401/403 且管理及目标用户状态未变；轮换后旧 key 401、新 key 可用 |
| metering_billing | 独立核对输入 token 数、可见输出重分词范围，响应 usage 与账目、汇总、余额和预占一致；benchmark key 也计费；两用户真实 Agent 父子账及幂等、价格、归属一致 |
| quota_recovery | 多 key 共享用户 RPM，超限 429 且 Retry-After；余额不足 402，不进入正常推理 |

真实部署检查不重复所有 CPU 故障注入；崩溃恢复由上述确定性测试验证。真实计费检查精确复核 chat-template 输入计数；输出用独立重分词后的可见文本做保守范围核验，容纳终止 token 和分词边界差异，不宣称精确获得引擎生成 token IDs。Agent 验收要求至少 2 次模型调用，子账数与 usage.model_calls 一致，没有固定只能 2 次或每次输出最多 160 的评测限制。参考实现自身选择两次各最多 160 tokens；这不是其他实现的评分门槛。父子账和响应字段见 [管理服务合同](credential-service.md)。正式性能另用独立 tokenizer 统计可见输出，不使用服务自报数量作为吞吐分子。

报告 credentials 项保存 `deployment_passed` 与 `live_checks`（每项 `id/passed/detail`），三个 `subchecks` 分别保存 `deployment_passed`。页面同时展示最终组分、真实部署依据和 15 项 CPU 原始结果。具体通过与否以本次控制端实测为准；已确认参考运行的凭据项目 15/15，见 [Daytona V5 实测记录](daytona-v5.md)，不能把该结果迁移到其他提交。

#### 报告中的通过、失败与未执行

`report.checks` 的 `id="credentials"` 项上限 15，`subchecks` 三组为 `key_lifecycle`、`metering_billing`、`quota_recovery`，每组上限 5。`cases` 按上述 test_id 保存 `id/title/group/executed/passed/detail`，不包含原文 key。

`executed=true` 且 `passed=true` 才是该项通过；执行后断言失败显示未通过。`executed=false` 或缺少记录显示未执行/无记录，并列出原因。未实现 `--contract-test` 或提交服务启动失败属于未完成交付，可得 0/15，不会因显示未执行而自动免扣分；控制器自身异常或失租的报告由平台标明中断，不能据此声称正式验收通过。公开自测只产出本地证据，不能提交分数或代替控制端测试。
