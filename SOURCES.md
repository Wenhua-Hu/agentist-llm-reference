# 复用与来源

参考工程的网关、部署、Agent、实验编排和测试为本项目实现。

- SGLang v0.5.2，Apache-2.0；源代码 commit `b0d25e72c401f37b55d689ddbf05b8c583afe854`。镜像 digest 固定于 Dockerfile。
- Qwen2.5-1.5B-Instruct，revision 固定于 README；遵循模型仓库许可证，不将权重提交到本仓库。
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
