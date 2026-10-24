# Agentist LLM-server — 模型服务参考实现

这是一份服务工程参考实现：SGLang 部署、持久化多租户凭据与实验账本、流式推理、FIFO 有界排队、取消/超时、工具 Agent、KV/前缀/Attention 实验与 CPU 测试。推理合同 llm-lab/1.0；管理合同 credential-service/1 已在独立 Daytona GPU 验证，包含真实 `/console` 与管理员 API。自动验收占 95 分、人工验收占 5 分，具体分项以平台当前项目任务书为准。历史报告不覆盖新增管理功能；当前 7B/V5 已完成标定与独立确认，自动 95/95，人工 5 分待验收。

## 当前参考版本与评分

请使用 [codex/credential-service-v5 分支](https://github.com/mrvgao/agentist-llm-reference/tree/codex/credential-service-v5)，默认分支仍是旧实现。正式复现再 checkout 平台任务书指定的完整 commit；不要把分支后续变化混入已固定的评测。

评分版本 `llm-lab/5.0` 共十项：重建 5、API 5、正确性 5、性能 30、Agent 5、KV 10、prefix 10、attention 10、多用户凭据/计费/配额 15，以及人工解释 5。管理服务的密钥生命周期、用量计费、配额恢复三组各 5 分，须同时满足 CPU 独立测试与真实 optimized SGLang 部署门槛；公开自测不能上传为成绩。完整 [评分细则与 15 项管理协议](docs/autograding.md) 说明每项通过、失败与未执行规则。

当前 GPU 实测固定源码为 `9961c0503c2eb3fa702794ef9281dd88f96d9456`。首轮标定及新环境独立确认已完成，见 [Daytona V5 实测记录](docs/daytona-v5.md)。后续仅文档/验收配置更新的提交与该实测源码 SHA 分开标识，不能声称后续新提交已被原样 GPU 跑过。

升级正在 staging 验证。性能门槛未标定或未独立确认时，30 分是“待评定”，不是学员失败 0 分；保留其他真实结果，暂不生成正式最终总分。控制器中断也不生成最终成绩。本轮已确认结果见 Daytona V5 实测记录；下列 A10 历史证据单独保留。

## CPU 接口检查与 OpenLabs 插件命令

本分支的 `.github/workflows/llm-soft.yml` 会在 `git push` 后，在 GitHub 托管的临时 Ubuntu CPU runner 中启动独立模型 fixture，再执行真实 `setup.sh --contract-test`、网关和 SQLite 账本。不会安装 SGLang、下载模型或申请平台 GPU。Actions Summary 显示逐项结果，`llm-soft-report` artifact 保留 `report.json` 和摘要。报告绑定完整 commit SHA；`pass / missing / invalid / blocked` 分别表示通过、缺接口、格式或行为错误、前置条件未满足。

安装正式版 **OpenLabs** 插件后，可让自己的 Agent 执行以下命令；`openlabs-test` 只用于内部测试，学员无需安装。项目名称为 `LLM-server`：

```text
evaluate LLM-server --soft
evaluate LLM-server result <返回的 job_id>
assistant LLM-server 我的 SSE / rotate / 计费检查为什么失败？
evaluate LLM-server --formal --repo <owner/repo> --commit <完整40位SHA>
```

手动 soft 使用本机 GitHub CLI（先 `gh auth login`），要求工作流存在于仓库默认分支，目标分支已经推送且 HEAD 与 SHA 一致；插件不会自动提交、push 或安装工作流。正式评测复用 `lab-login`、网页已绑定仓库、cohort-2 资格与现有 Realtime 队列；只在明确请求 `--formal` 时使用 GPU 预算和正式次数。自动 push 检查不改变 95+5 分成绩，不证明模型质量、吞吐、TTFT、TPOT、KV Cache 或 GPU 优化达标。CPU 报告属于可修改的自测证据，正式评分仍由平台独立评测器产生。

学员任务说明及当前检查器下载见 [项目任务书](https://agentist.org/lab/llm-lab/taskbook)。内部验收使用 [测试环境任务书](https://test.lab.parallight.ai/lab/llm-lab/taskbook)，测试环境成绩不自动发布到正式环境。不要向工作流提供平台凭据、Daytona/Lambda API Key 或生产密钥。公开仓库默认可查看 Actions 日志；脚本只使用一次性测试密钥。

## 历史 A10 彩排（不作为当前验收）

历史 llm-lab/2.0 自动评测 **50/50**，576/576 正式请求正确，Agent 10/10；空环境重建约 **197 秒**。并发 8 的吞吐从 66.88 提升到 451.57 tokens/s。FlashAttention 数值检查通过，前缀缓存命中但小负载延迟收益很小；BF16 KV 全 logits 检查失败，负结果保留，FP32 对照通过，结果单独报告。

[实测报告与图表](docs/performance.md) · [项目验收证据索引](docs/acceptance.md) · [原始数据](results/)

## 从干净 Daytona GPU Linux 启动

前置边界：Linux x86_64、NVIDIA RTX 4090 驱动可用、Git、Python 3.12/venv、外网。`setup.sh` 在项目私有 `.venv` 安装依赖，不要求 Docker 或 systemd，不修改 GPU 驱动。若缺少与所选 Python 对应的 `Python.h` 或 C/C++ 编译器，setup 会使用非交互 sudo/apt 安装匹配的 Python 开发头文件及 build-essential；基础环境须允许此操作。引擎关键版本见 `requirements-runtime.lock`，完整 Linux/Python 3.12 包版本见 `requirements-native-linux-py312.lock`；默认一次安装网关、引擎和分析依赖，随后执行 pip check 与完整版本集合核对，复现边界见[运行时锁说明](docs/native-runtime-lock.md)；可复用平台下载缓存，但项目配置与代码从指定 Git commit 重建。若平台提供已固定版本的引擎解释器，可以显式设置 `SGLANG_PYTHON=/path/to/python`，setup 会检查版本；网关仍使用私有 venv。干净安装和预置运行时必须分别记录，不能混称为同一种冷启动。历史 Lambda 路径保留在 `setup-lambda.sh`，不代表当前配置已在 A10 重测。

Triton / FlashInfer 即使已安装 wheel，也可能在第一次推理时 JIT 编译 Python 扩展，因此必须具备匹配解释器版本的 `Python.h` 和 C/C++ 编译工具。setup 只使用系统已有的官方 Ubuntu 软件源安装这些依赖，保留签名验证并跳过无关 CUDA 第三方源；网络超时 20 秒、重试 1 次，update/install 总时限分别为 180/240 秒。安装进度会显示在评测终端，失败会返回非零，避免网络问题无限占用 GPU。直接调用 `.venv/bin/python` 不会自动激活 PATH；setup 和实验入口会显式加入实际引擎解释器的 bin、网关 venv/bin 和 CUDA_HOME/bin，并检查 cc/c++、ninja、nvcc。工具链路径记录在 `.runtime/runtime.json`，无需手动 source activate。

```bash
git clone --branch codex/credential-service-v5 https://github.com/mrvgao/agentist-llm-reference.git
cd agentist-llm-reference
export HOST=127.0.0.1 PORT=8000
export API_KEY="$(openssl rand -hex 32)"
export ADMIN_API_KEY="$(openssl rand -hex 32)"
export KEY_HASH_SECRET="$(openssl rand -hex 32)"
export DATA_DIR="$PWD/.data"
# 将这些变量保存在受限的环境配置中；重启复用相同 KEY_HASH_SECRET / DATA_DIR。
export MODEL_ID=Qwen/Qwen2.5-7B-Instruct
export MODEL_REVISION=a09a35458c702b33eeacc393d103063234e8bc28
export BENCH_PROFILE=baseline
export HF_HOME="$HOME/.cache/huggingface"
bash setup.sh
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS -H "Authorization: Bearer $API_KEY" http://127.0.0.1:8000/info
```

`setup.sh` 等待就绪后退出；服务由本项目 supervisor 管理，记录 PID 与 Linux 进程启动标识，日志在 `.runtime/`。相同配置幂等；`BENCH_PROFILE=optimized bash setup.sh` 停本项目旧服务再启动优化配置。密钥由环境传递；数据库只保存 HMAC 摘要和掩码，不能保存明文。`API_KEY` 是普通 metered benchmark 用户，不拥有管理权限。`ADMIN_API_KEY` 与它必须不同；未提供管理员凭据时管理 API 关闭。

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"Explain KV caching in one sentence."}],"max_tokens":96,"stream":true}'
```

## 代码入口

- `setup.sh` / `scripts/native_setup.py` / `server/supervisor.py`：可重建、幂等、进程生命周期。
- `server/app.py` / `admission.py` / `streaming.py` / `backend.py`：公开合同、排队、真实 SGLang 转发、取消。
- `control_plane/storage.py`：SQLite 迁移、HMAC 密钥、版本价格、事务预占/结算、跨密钥配额与审计；`api.py`：严格管理接口；`meter.py`：可信输入计数、恢复、数据库进程所有权。
- `console/index.html`：用户、密钥、额度、价格、账单与审计管理；`migrations/` / `configs/pricing.json`：schema 和初始实验价格。
- `scripts/seed_demo.py` / `scripts/check_control_plane.py`：合成用户与公开自测，不写入平台正式成绩。
- `agent/`：模型规划 → 白名单 lookup_order → 模型最终回答；不从题目绕过模型生成答案。
- `configs/`：baseline 1 个 running request / 8192 KV tokens / Triton / prefix off / CUDA Graph off；optimized 32 / 16384 / FlashInfer / prefix on / CUDA Graph on。两组固定模型、BF16、4096 上下文和网关 32 slots/64 queue。两组显式使用 `sampling_defaults=openai`，请求温度固定 0，避免继承模型文件里的重复惩罚；见[采样策略](docs/sampling-policy.md)。总 profile 比较不能归因到单项；用 `MAX_RUNNING_REQUESTS`、`PREFIX_CACHE`、`ATTENTION_BACKEND` 做受控消融。
- `experiments/`：KV Cache、RadixAttention、SDPA math/flash、服务压测、GPU 取消验证与 SVG。
- `tests/` / `.github/workflows/ci.yml`：只有 CPU 测试，不申请 GPU。
- `docs/` / `results/`：架构、运维、性能结论与原始证据。见 `docs/performance.md` 实测状态。

## 多用户管理与实验账本

浏览器通过 SSH tunnel 打开 `/console`，输入独立的管理员凭据即可创建用户、调整共享配额/余额、发放与轮换密钥、查询账单。页面不保存管理员密码到 localStorage/cookie；新 API key 只在创建或轮换时显示。所有金额为整数 microUSD，初始实验价格为输入 2、输出 6 每 token，不代表市场定价，不发生真实支付。

[完整管理 API、持久化、故障与自测说明](docs/credential-service.md)。CPU 模拟测试不启动模型；live 自测只调用已运行的服务，不申请 GPU：

```bash
.venv-test/bin/python scripts/check_control_plane.py --suite mock \
  --out results/control-plane/mock-run-1.json
.venv/bin/python scripts/check_control_plane.py --base-url http://127.0.0.1:8000 \
  --suite live --out results/control-plane/live-run-1.json
```

`mock` 覆盖 1–11 场景，真实 GPU 场景明确 skipped；`live` 验证两个用户的流式、非流式及 Agent 账单，并明确跳过需要受控故障/用量的场景。任何 skipped 不计通过，公开自测没有正式评分权。

## 显式运行实验

当前服务固定 `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`、BF16；baseline/optimized 的静态显存比例均为 0.8，已在本轮两台独立 GPU 验证；这项配置不保证其他模型或负载同样通过。KV 原理实验独立固定 `Qwen/Qwen2.5-1.5B-Instruct` revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`、FP32（禁用 TF32），保留 0.001 最大绝对误差与 100% top-1 门槛；7B 的 FP32 权重超过 24 GiB，不能在该 GPU 上直接替换原理模型。prefix 使用服务 7B；attention 使用合成 Q/K/V，7B 仅标记服务上下文，不加载模型权重。入口默认按 case 选择模型，显式错误模型/revision或 KV dtype 会被拒绝，不能静默改写 manifest。

先运行一次 setup 安装固定环境。以下每个命令会停服务，独占 GPU；不会偷偷与服务同时加载两个模型。每个输出目录只能写一次，重跑需换目录。

```bash
MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct \
MODEL_REVISION=989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  bash experiments/run.sh --case kv --out results/kv/run-1
bash experiments/run.sh --case attention --out results/attention/run-1
bash experiments/run.sh --case prefix --out results/prefix/run-1
bash setup.sh # 显式恢复服务
.venv/bin/python -m experiments.cancellation
```

实验失败返回非零并保存 failure.json；不会静默换后端。完整记录 manifest/raw/summary/SVG，profiler 大文件在实验目录生成，发布时附 SHA256。优化服务使用 FlashInfer；SDPA FlashAttention 是独立机制实验，不宣称二者是同一个实现。服务性能必须通过外部请求实测。

## 对最终服务运行公开自测

这组工具与平台使用相同的固定输入长度、正确性检查和指标定义，结果用于调试，不会写入正式成绩，也不会申请 GPU。必须先启动自己的服务。默认 `benchmarks/acceptance.json` 是已经独立确认的 7B v5 验收合同（prose padding + JSON 模式，长输入带记录编号并保持原顺序）。下例使用这个已确认文件；`benchmarks/acceptance-v5.pending.json` 保留无阈值历史候选，只供对照。模型身份不符时会拒绝运行。门槛为空时绝不算通过；候选门槛只有经过另一台全新 GPU、相同源码 commit、不同随机 seed 的独立确认后，才能用于正式计分。验收必须完整包含 short/long × 并发 8/16 四项目标，缺项不得按剩余项目折算满分。

```bash
# 默认 setup 的项目 venv 已包含所需依赖；API_KEY 与服务启动时相同。
BENCH_PROFILE=baseline bash setup.sh
.venv/bin/python -m benchmarks.serving_benchmark \
  --config benchmarks/acceptance.json \
  --base-url http://127.0.0.1:8000 --profile baseline \
  --seed 20261017 --out results/self-test/run-1
BENCH_PROFILE=optimized bash setup.sh
.venv/bin/python -m benchmarks.serving_benchmark \
  --config benchmarks/acceptance.json \
  --base-url http://127.0.0.1:8000 --profile optimized \
  --seed 20261017 --out results/self-test/run-1
```

每次新的比较使用新的 seed 和目录；同一轮两组 profile 使用相同 seed。工具先核对 `/info` 中的真实 profile、模型、revision、精度及 SGLang 引擎；不会把 baseline 误标为 optimized。若使用平台预置 `SGLANG_PYTHON`，可以使用该解释器运行自测。

- `benchmarks/acceptance.json`：硬件、固定模型、负载、并发和标定后的性能门槛。
- `benchmarks/serving_benchmark.py`：构造精确 512/1024 chat-template 输入 token 的短/长 JSON 任务，运行并发 1/4/8/16、三轮采样；校验答案的字段、类型和值；用固定 tokenizer 独立计数，不相信服务自报 token 数。
- `benchmarks/measurement.py`：SSE 事件解析与分位数计算。
- 输出 `*-serving-raw.jsonl` 保存逐请求结果，`*-serving-summary.json` 保存吞吐、TTFT/TPOT/E2E p95、成功率和正确率；有已标定 SLO 时同时给出 goodput。

测量版本 `controller-sse/2` 区分 `request_timeout`、`upstream_timeout` 和其他流失败；这些请求始终计入失败分母，部分输出不计有效吞吐。历史没有明确错误码的失败不能重新标成超时。

SSE 一次事件可能包含多个 token；`event_gap_p95_ms` 表示流式事件间隔，不冒充逐 token 的精确 ITL。公开自测脚本和输出都可被修改，因此正式分数只能由平台在干净环境独立运行后生成。

## CPU 测试

```bash
python3 -m venv .venv-test
.venv-test/bin/pip install -r requirements.lock pytest==8.4.1 pytest-asyncio==1.1.0
.venv-test/bin/python -m pytest -q
```

正式评测只通过平台“评测”显式触发，clone 指定完整 commit，再执行 setup 和独立测量。push/CI 不触发 GPU。参考实现不包含 Daytona、数据库、平台服务密钥。

## KV 数值诊断

历史 A10 和首轮 Daytona 实测均发现 BF16 cache on/off 的全词表 logits 差异超过预设 atol=0.15/rtol=0.02；负结果原样保留，不修改答案或放宽容差。当前 KV 数学等价实验默认使用 FP32，并关闭 TF32，使用更严格的 atol=0.001/rtol=0.0001；最终 SGLang 服务仍固定 BF16。FP32 原理实验不代替 BF16 服务正确性与性能验收。当前入口只接受固定 1.5B FP32；旧版本支持的 `KV_DTYPE=bfloat16` 诊断不适用于此提交，历史 BF16 证据应按对应旧源码 commit 阅读与复现。

FP32 对照使用预先固定的 atol=0.001/rtol=0.0001，两分支同为 FP32；A10 实测通过，最大误差约 0.00005。其结果不能替代 BF16 验证。两种精度均报告 top-1 agreement 与全词表误差。正式服务仍固定 BF16。
