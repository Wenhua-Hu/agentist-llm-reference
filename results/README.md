# 实测数据导航

- `formal/`：第二轮正式评分，工作负载 extraction-typed/1，自动分 50/50；包含原始请求、预热、题目和环境声明。
- `confirmation/`：第三轮正式复测，自动分再次 50/50；包含另一组随机工作负载与取消日志。
- `kv-diagnostics/{bf16,fp32}/`：同机独立精度控制；BF16 原阈值失败，FP32 预设严格阈值通过。
- `first-attempt/`：首轮 34/50 的完整记录，保留 JSON 类型题目歧义导致的失败。
- `kv/rehearsal/`：BF16 cache on/off 对照。计时完成，但全词表数值检查未通过；failure.json 不应删除。
- `attention/rehearsal/`：强制 math/flash 对照；trace.json.gz 解压后的 SHA256 与同目录校验记录匹配。
- `prefix/rehearsal/`：on/off × shared/unique 的 96 条请求与真实 cache 命中计数。
- `cancellation/`：20 次真实流式连接取消及前后引擎指标。
- `analysis/`：由上述数据生成的图，原始数据没有被图表覆盖。
- `provenance.json`：运行编号、代码 SHA、目录映射、回收状态和成本估算。

这些是 Lambda GPU 实测，不是模拟结果。CPU/mock 测试不进入这些性能数据。代码版本、精度或工作负载不同的试验分别保存，不能拼成一个优化收益。

从仓库根目录运行 `python3 -m experiments.verify_evidence` 可检查数据一致性，运行 `python3 -m experiments.overview` 重建概览图。分析依赖见 requirements-analysis.lock，不需要 GPU。
