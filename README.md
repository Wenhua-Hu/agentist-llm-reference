# Agentist Project 2 — LLM service reference implementation

这是一份完整参考工程：固定模型部署、鉴权网关、流式推理、FIFO 有界排队、取消/超时、工具 Agent、KV/前缀/Attention 实验、CPU 测试与实测报告。接口合同 llm-lab/1.0；评分合同 llm-lab/2.0。

## 从干净 Lambda Linux 启动

前置边界：Ubuntu x86_64、NVIDIA A10 驱动可用、Git、sudo、外网。Docker 缺失时 setup 安装；Lambda 镜像需具有 NVIDIA container runtime。不修改 GPU 驱动。所有项目 Python/引擎依赖来自 Dockerfile 的不可变镜像 digest + requirements.lock；模型缓存首次为空。

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

`setup.sh` 等待就绪后退出；服务在受管理容器中运行，日志在 `.runtime/`。相同配置幂等；`BENCH_PROFILE=optimized bash setup.sh` 停本项目旧服务再启动优化配置。密钥由环境传递，真实密钥不入库。

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"Explain KV caching in one sentence."}],"max_tokens":96,"stream":true}'
```

## 代码入口

- `setup.sh` / `Dockerfile` / `server/supervisor.py`：可重建、幂等、进程生命周期。
- `server/app.py` / `auth.py` / `admission.py` / `streaming.py` / `backend.py`：公开合同、鉴权、排队、真实 SGLang 转发、取消。
- `agent/`：模型规划 → 白名单 lookup_order → 模型最终回答；不从题目绕过模型生成答案。
- `configs/`：baseline 1 个 running request / 8192 tokens，optimized 8 / 16384。模型、BF16、4096 上下文、Triton attention、prefix off 相同。CUDA Graph 两组均关闭，减少部署编译并独立观察批处理。
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
sudo docker exec agentist-llm-reference python3 -m experiments.cancellation
```

实验失败返回非零并保存 failure.json；不会静默换后端。完整记录 manifest/raw/summary/SVG，profiler 大文件在实验目录生成，发布时附 SHA256。服务使用 Triton，并不宣称其等同于 SDPA 的 FlashAttention 后端。

## CPU 测试

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock pytest==8.4.1 pytest-asyncio==1.1.0
.venv/bin/python -m pytest -q
```

正式评测只通过平台“评测”显式触发，clone 指定完整 commit，再执行 setup 和独立测量。push/CI 不触发 GPU。参考实现不包含 Lambda、数据库、平台服务密钥。
