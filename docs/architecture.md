# 架构与实现边界

客户端 → SSH loopback tunnel → FastAPI 鉴权/请求 ID → FIFO gate → SGLang HTTP → tokenizer/scheduler → GPU。仅一个 Uvicorn worker，gate 对整个网关进程有效；不能直接增加 worker 数，否则限制会变成每进程。

队列同时限制 active 和 pending，FIFO 将槽位所有权先转交再唤醒；取消与转交竞争时归还槽位。每个请求有总 deadline，包含排队和上游。streaming 只有 finish_reason、usage、DONE 完整时才记成功；空 role 事件不是内容 token。取消关闭上游流，并使用同一个 rid 显式调用 SGLang abort_request。异步清理任务在服务退出时汇合，abort 失败单独计数。

模型输出/usage 直接来自引擎，不替换成固定答案。Agent 第一次模型调用只收到任务和工具描述，不收到订单事实；白名单工具执行后，第二次模型调用从真实工具结果回答。输出不正确时不偷偷用工具数据改写。最大两次模型调用与总超时限制防止无界循环。

鉴权仅健康检查豁免。SGLang 引擎只监听 loopback；网关默认 loopback，可通过 HOST=0.0.0.0 接入 Daytona 受控端口代理。API key 不打印、不进 /info。正式测试端应独立控制认证和评分，不能信任学员提交的成绩。原生 supervisor 管理进程，不要求 Docker 或 systemd。

模型权重 ≈ P×b_w。逻辑 KV = 2×L×Hkv×dh×bkv×ΣT_i。引擎预分配池、PyTorch reserved 和 nvidia-smi 进程显存包含不同对象，不能直接互相等同。PagedAttention 改善分配/碎片与复用，不减少这个逻辑张量元素数量。
