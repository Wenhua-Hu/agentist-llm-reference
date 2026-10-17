# 启停与排错

1. 先 `nvidia-smi`；驱动失败由平台处理，不升级驱动。
2. `bash setup.sh` 的 stdout 是安装/就绪进度；`.runtime/engine.log`、`gateway.log` 是长期进程日志。不可发布 `docker inspect` 的完整环境，那里有 API_KEY。
3. 只有 setup 成功并且 `/healthz` 返回 ready 才可调用。503 看引擎日志，401 检查 Bearer，429 检查队列，504 检查总 deadline。SSE error 或缺 DONE 是失败，不算成功响应。
4. 配置改变重跑 setup。停止仅 `sudo docker stop agentist-llm-reference`，不能 killall python。
5. `sudo docker exec agentist-llm-reference python3 -m experiments.cancellation` 验证 20 次取消。日志在 `.runtime/cancellation`，同时保留引擎 running/used token 指标与网关 active/queue/abort 计数。
6. 实验停服务后不自动恢复，下一步显式 `bash setup.sh`。保存 results 后 git commit/push，模型/密钥/环境不入 Git。
7. GPU 计费由 Lambda terminate 才结束；Docker stop 或操作系统 shutdown 不等于释放。使用平台结束按钮并确认云端回收。
8. 首次模型下载网络失败：保留失败日志，修复网络后重跑 setup；已下载缓存会复用，该次不能再声称是冷启动。

当前脚本针对专属 Lambda VM，不给网页服务器部署推理环境。正式评分由独立控制器执行，不信任仓库自行上报的分数。
