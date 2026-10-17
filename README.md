# Agentist Project 2 — LLM service reference implementation

这是一份完整参考工程：固定模型部署、鉴权网关、流式推理、FIFO 有界排队、取消/超时、工具 Agent、KV/前缀/Attention 实验、CPU 测试与实测报告。接口合同 llm-lab/1.0；当前评分合同 llm-lab/3.0：自动 95 分，人工验收仅 5 分（工程与优化解释）。KV / prefix / attention 各 15 分由平台实际运行后自动评分。具体调用与字段见[自动评分合同](docs/autograding.md)。

## 已完成的真实 A10 彩排

历史 llm-lab/2.0 自动评测 **50/50**，576/576 正式请求正确，Agent 10/10；空环境重建约 **197 秒**。并发 8 的吞吐从 66.88 提升到 451.57 tokens/s。FlashAttention 数值检查通过，前缀缓存命中但小负载延迟收益很小；BF16 KV 全 logits 检查失败，负结果保留，FP32 对照通过，结果单独报告。

[实测报告与图表](docs/performance.md) · [项目验收证据索引](docs/acceptance.md) · [原始数据](results/)

## 从干净 Daytona GPU Linux 启动

前置边界：Linux x86_64、NVIDIA RTX 4090 驱动可用、Git、Python 3.12/venv、外网。`setup.sh` 在项目私有 `.venv` 安装依赖，不使用 Docker、sudo 或 systemd，不修改 GPU 驱动。引擎关键版本见 `requirements-runtime.lock`；可复用平台下载缓存，但项目配置与代码从指定 Git commit 重建。若平台提供已固定版本的引擎解释器，可以显式设置 `SGLANG_PYTHON=/path/to/python`，setup 会检查版本；网关仍使用私有 venv。干净安装和预置运行时必须分别记录，不能混称为同一种冷启动。历史 Lambda 路径保留在 `setup-lambda.sh`，不代表当前配置已在 A10 重测。

```bash
git clone https://github.com/mrvgao/agentist-llm-reference.git
cd agentist-llm-reference
export HOST=127.0.0.1 PORT=8000
export API_KEY="$(openssl rand -hex 32)"
export MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
export MODEL_REVISION=989aa7980e4cf806f80c7fef2b1adb7bc71aa306
export BENCH_PROFILE=baseline
export HF_HOME="$HOME/.cache/huggingface"
bash setup.sh
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS -H "Authorization: Bearer $API_KEY" http://127.0.0.1:8000/info
```

`setup.sh` 等待就绪后退出；服务由本项目 supervisor 管理，记录 PID 与 Linux 进程启动标识，日志在 `.runtime/`。相同配置幂等；`BENCH_PROFILE=optimized bash setup.sh` 停本项目旧服务再启动优化配置。密钥由环境传递，真实密钥不入库。

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"Explain KV caching in one sentence."}],"max_tokens":96,"stream":true}'
```

## 代码入口

- `setup.sh` / `scripts/native_setup.py` / `server/supervisor.py`：可重建、幂等、进程生命周期。
- `server/app.py` / `auth.py` / `admission.py` / `streaming.py` / `backend.py`：公开合同、鉴权、排队、真实 SGLang 转发、取消。
- `agent/`：模型规划 → 白名单 lookup_order → 模型最终回答；不从题目绕过模型生成答案。
- `configs/`：baseline 1 个 running request / 8192 KV tokens / Triton / prefix off / CUDA Graph off；optimized 32 / 16384 / FlashInfer / prefix on / CUDA Graph on。两组固定模型、BF16、4096 上下文和网关 32 slots/64 queue。总 profile 比较不能归因到单项；用 `MAX_RUNNING_REQUESTS`、`PREFIX_CACHE`、`ATTENTION_BACKEND` 做受控消融。
- `experiments/`：KV Cache、RadixAttention、SDPA math/flash、服务压测、GPU 取消验证与 SVG。
- `tests/` / `.github/workflows/ci.yml`：只有 CPU 测试，不申请 GPU。
- `docs/` / `results/`：架构、运维、性能结论与原始证据。见 `docs/performance.md` 实测状态。

## 显式运行实验

先运行一次 setup 安装固定环境。以下每个命令会停服务，独占 GPU；不会偷偷与服务同时加载两个模型。每个输出目录只能写一次，重跑需换目录。

```bash
bash experiments/run.sh --case kv --out results/kv/run-1
bash experiments/run.sh --case attention --out results/attention/run-1
bash experiments/run.sh --case prefix --out results/prefix/run-1
bash setup.sh # 显式恢复服务
.venv/bin/python -m experiments.cancellation
```

实验失败返回非零并保存 failure.json；不会静默换后端。完整记录 manifest/raw/summary/SVG，profiler 大文件在实验目录生成，发布时附 SHA256。优化服务使用 FlashInfer；SDPA FlashAttention 是独立机制实验，不宣称二者是同一个实现。服务性能必须通过外部请求实测。

## CPU 测试

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock pytest==8.4.1 pytest-asyncio==1.1.0
.venv/bin/python -m pytest -q
```

正式评测只通过平台“评测”显式触发，clone 指定完整 commit，再执行 setup 和独立测量。push/CI 不触发 GPU。参考实现不包含 Daytona、数据库、平台服务密钥。

## KV 数值诊断

首次 A10 实测发现 BF16 cache on/off 的全词表 logits 差异超过预设 atol=0.15/rtol=0.02；负结果原样保留，默认命令仍按原阈值返回失败，不修改答案或事后放宽容差。可独立运行 FP32、禁用 TF32 的控制实验定位误差来源：

```bash
KV_DTYPE=float32 bash experiments/run.sh --case kv --out results/kv-fp32/rehearsal
```

FP32 对照使用预先固定的 atol=0.001/rtol=0.0001，两分支同为 FP32；A10 实测通过，最大误差约 0.00005。其结果不能替代 BF16 验证。两种精度均报告 top-1 agreement 与全词表误差。正式服务仍固定 BF16。
