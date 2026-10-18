# 复用与来源

参考工程的网关、部署、Agent、实验编排和测试为本项目实现。

- 当前 native SGLang v0.5.9，由 requirements-runtime.lock 固定完整依赖闭包。以下源码行号和旧 Docker 镜像来自历史 SGLang v0.5.2，Apache-2.0；源代码 commit `b0d25e72c401f37b55d689ddbf05b8c583afe854`。镜像 digest 固定于 Dockerfile。
- 候选服务：[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)，revision `a09a35458c702b33eeacc393d103063234e8bc28`，BF16。
- KV 原理实验：[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，FP32。两者遵循模型仓库许可证，不将权重提交到本仓库。历史性能记录保留原模型身份。
- PyTorch / Transformers / FastAPI / httpx 等保持各自许可证；Python依赖及引擎镜像锁定。
- `experiments/measurement.py` 复用 Agentist Project 2 提供的测量脚手架，保留原始客户端计时、SSE和质量检查语义；`serving_bench.py` 为本项目编排。

## 机制定位（固定 SGLang commit）

完整前缀匹配、插入和淘汰：
https://github.com/sgl-project/sglang/blob/b0d25e72c401f37b55d689ddbf05b8c583afe854/python/sglang/srt/mem_cache/radix_cache.py

- `match_prefix` 158；`insert`195；`evict`294；`inc_lock_ref`319；`dec_lock_ref`333。
- lock_ref保护活跃请求引用的节点；引用归零后成为可淘汰对象。释放请求不等于所有前缀缓存立即从GPU进程显存消失。

分配/释放：`python/sglang/srt/mem_cache/allocator.py` 的 alloc/free 与 `memory_pool.py` 的 request slot分配。
请求取消：`http_server.py`881 接收rid；`tokenizer_manager.py`845转发AbortReq；scheduler清理请求并释放缓存引用。
调度：`python/sglang/srt/managers/scheduler.py` 的 get_new_batch_prefill / update_running_batch，以及 scheduler_output_processor.py 中的完成/abort处理。engine启动日志的running request数量、token预算与测试取消后的占用恢复作为运行证据。

源码定位只解释机制；不能替代GPU实测或cached_tokens/profiler证据。

## nano-vLLM 调度器的预算与抢占

固定 commit `bb823b3e06983d71485a8e1f23715ebd87d98ef8`：
https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py

`schedule()` 的 prefill 分支同时受 max_num_seqs 与 max_num_batched_tokens 限制。remaining = budget - scheduled_tokens；前缀命中减少待计算token，第一请求可以chunked prefill，不能把完整prompt长度都算成新计算量。decode每轮为每个活跃sequence调度1个token；can_append失败时从running尾部抢占，再释放块并回到waiting。postprocess完成时立即移除，下一轮新请求可以进来，这才是连续批处理，而非等整个批次都结束。

`block_manager.py` 的 allocate增加ref_count；deallocate递减，仅归零才回收block。hash_blocks用于前缀块复用。SGLang的radix tree按token前缀组织，nano-vLLM按块hash组织；两者不能只因都复用KV就声称实现完全相同。

本工程的两个profile改变SGLang真实调度预算，CPU网关容量保持相同；源码用于解释机制，不将客户端并发增加冒充引擎算法修改。

容器运行时缺失时采用 NVIDIA 官方安装流程，固定 toolkit 1.20.1-1；已存在的 Lambda runtime 保持原状，记录在主机探针中。
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
