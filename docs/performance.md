# 性能报告

实测尚未执行。不得把 CPU mock 测试或预期收益当作 GPU 结果。彩排完成后在此记录独立评分器输出与三项优化实测，并保留失败尝试。

## 对照设计

- 连续批处理：固定模型/revision/BF16/Triton/prefix off/CUDA Graph off，只改 running-request 和 token budget；外部并发1/4/8，每档3×32，计576正式请求。
- KV：128/512 token 输入，离线固定32 token continuation，两分支回放相同位置；预热后交替3次；decode计时不含tokenize/加载/prefill。
- RadixAttention：512 token 前缀，8条同长度请求；shared/unique×on/off×3，kernel预热后清缓存；保留冷/热/后续标签和真实cached_tokens。
- FlashAttention：BF16、B1/H8/d64、N128/512/1024，math/flash各5预热+3×10计时；预设atol/rtol=.02，强制后端、profiler证明。

## 测量限制

TTFT/E2E由客户端计时；TPOT是首末内容事件间隔除以真实usage输出token-1的代理，不是逐tokenITL。正式平台尚未独立tokenize核对usage。报告必须分别展示成功率、正确率、SLO goodput，不以吞吐单指标得出结论。微基准Flash收益不能直接当成端到端收益。
