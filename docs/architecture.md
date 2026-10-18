# 架构与实现边界

客户端 → SSH loopback tunnel → FastAPI 鉴权/请求 ID → FIFO gate → SGLang HTTP → tokenizer/scheduler → GPU。仅一个 Uvicorn worker，gate 对整个网关进程有效；不能直接增加 worker 数，否则限制会变成每进程。

队列同时限制 active 和 pending，FIFO 将槽位所有权先转交再唤醒；取消与转交竞争时归还槽位。每个请求有总 deadline，包含排队和上游。streaming 只有 finish_reason、usage、DONE 完整时才记成功；空 role 事件不是内容 token。取消关闭上游流，并使用同一个 rid 显式调用 SGLang abort_request。异步清理任务在服务退出时汇合，abort 失败单独计数。

模型输出/usage 直接来自引擎，不替换成固定答案。Agent 第一次模型调用只收到任务和工具描述，不收到订单事实；白名单工具执行后，第二次模型调用从真实工具结果回答。输出不正确时不偷偷用工具数据改写。最大两次模型调用与总超时限制防止无界循环。

健康检查与不含敏感数据的 `/console` 登录壳公开；其他请求分别使用普通租户密钥或独立管理员凭据鉴权。SGLang 引擎只监听 loopback；网关默认 loopback，可通过 HOST=0.0.0.0 接入 Daytona 受控端口代理。API key 不打印、不进 /info。正式测试端应独立控制认证和评分，不能信任学员提交的成绩。原生 supervisor 管理进程，不要求 Docker 或 systemd。

模型权重 ≈ P×b_w。逻辑 KV = 2×L×Hkv×dh×bkv×ΣT_i。引擎预分配池、PyTorch reserved 和 nvidia-smi 进程显存包含不同对象，不能直接互相等同。PagedAttention 改善分配/碎片与复用，不减少这个逻辑张量元素数量。

多租户管理的事务与恢复边界见 [credential-service/1](credential-service.md)。SGLang 不承担用户账本；网关先用固定 chat template 计算输入 token，再在 SQLite 事务内预占余额与 token 额度。流式最终 usage 在向客户端发出前落盘，DONE 后结算；取消释放执行名额，未知账目继续占用余额直到审计对账。


## 测试边界与正式评分

当前参考源码位于 [codex/credential-service-v5](https://github.com/mrvgao/agentist-llm-reference/tree/codex/credential-service-v5)；正式运行固定平台任务书中的 commit，默认分支不是当前管理服务实现。`llm-lab/5.0` 的十项分值是重建 5、API 5、正确性 5、性能 30、Agent 5、KV/prefix/attention 各 10、多用户服务 15、人工解释 5；合计自动 95、人工 5。

控制端对管理服务执行三组共 15 个确定性检查：密钥生命周期、用量计费、配额恢复各 5 分。`setup.sh --contract-test` 只把模型引擎替换成外部 fixture，鉴权、账本、事务和持久化继续走同一实现；不能绕过计费或返回特殊通过结果。fixture 不归提交项目管理，`--stop` 只能停止本项目进程；`DATA_DIR/gateway.pid` 必须指向实际网关进程。控制端验证端口所有权后注入崩溃，分别测试未知 usage 的恢复和已持久化 terminal usage 的恰好一次结算。

每个管理服务 5 分组还须通过对应真实 optimized SGLang 网关检查，否则该组为 0，CPU 原始 case 仍保留。独立平台测试与可修改的 `scripts/check_control_plane.py` 公开自测分离；公开自测结果不具备正式评分权，也不能用 mock 替代真实 GPU。详见 [完整 15 项协议、部署门槛与证据限制](autograding.md)。

性能 30 分须由完整实测、冻结配置与新环境独立确认共同支撑。未标定或未确认显示待评定，不能判作学员失败或生成正式最终分；控制器中断报告只用于排查。本轮 v5 已完成独立 GPU 确认，自动 95/95、人工 5 分待验收；结果严格绑定 [实测运行与源码 SHA](daytona-v5.md)，不迁移到其他提交。
