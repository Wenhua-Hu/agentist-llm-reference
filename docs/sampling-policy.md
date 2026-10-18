# 显式采样策略与数值复制

baseline 和 optimized 都通过 `--sampling-defaults openai` 启动 SGLang；`/info.config.sampling_defaults` 报告 `openai`。`/info.config` 同时报告该策略对应的 repetition_penalty、top_p、top_k 和 request_temperature_default；它们是启动配置声明，评测还应核对引擎实际参数。项目评测请求明确发送 `temperature=0`，普通 API 请求仍允许显式调整温度。SGLang 0.5.9 在此策略下使用中性的缺省值：repetition_penalty=1.0、top_p=1.0、top_k=-1。用户消息和输出内容由网关原样转发；网关不会修补数字或把模型答案替换成标准答案。

当前候选服务 `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28` 的官方 generation_config.json 包含 repetition_penalty=1.05、temperature=0.7、top_p=0.8、top_k=20，同样必须显式使用 openai 中性策略。KV 原理实验独立固定 1.5B FP32，不能将其数值对照误报成 7B 服务结果。

历史背景：这项配置避免一个已确认的隐式差异：当时固定模型 `Qwen/Qwen2.5-1.5B-Instruct` revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` 的 generation_config.json 包含 repetition_penalty=1.1、temperature=0.7、top_p=0.8、top_k=20；SGLang 0.5.9 默认 `sampling_defaults=model`，显式请求 temperature=0 只覆盖温度，不会自动关闭其他模型缺省参数。

Daytona 运行 `39065b3b-fa51-4078-8b5c-ea322efa9fa8`、源码 `ce1fa35c2592381a201b5ccf4f2a599d10a8c93b` 的只读检查已确认运行中引擎采用 `model`，本地缓存重复惩罚为 1.1。该轮原始数字复制错误、延迟和吞吐均保留，不能修改成中性采样结果。错误是否主要由重复惩罚导致，需要新环境、新 commit 的实测对照；目前不能声称因果已经证明。

中性采样策略本身不降低正确性门槛。当前 7B 是独立模型候选，需按新 revision、显式 JSON 模式与版本化输入重新验证及标定；不能复用旧 1.5B 的性能结论。若将来要改变采样策略，必须更新任务书、配置和验收版本，不能在 baseline/optimized 之间偷偷使用不同策略。

网关在响应开始前遇到 HTTPX 连接、读取、写入或连接池超时，会返回 504；其他连接故障仍返回 503。已经开始的 SSE 流遇到超时或故障会发送 error 且不发送 DONE，避免把半截输出算作成功。超时本身不证明是容量不足，评测需结合负载、引擎状态和恢复情况判断。

流中途错误还带有 `error.code`：网关总截止时间触发为 `request_timeout`，HTTPX 自身超时为 `upstream_timeout`，流格式错误、提前结束或其他传输故障为 `stream_failure`。只有本地实际捕获的超时异常能产生前两种错误码；引擎发送的未知 error 不会被重新解释为超时。HTTP 状态此时仍为 200，但请求应记录为失败。历史通用错误不能追溯改成超时，更不能仅凭超时码断言 GPU 容量不足。
