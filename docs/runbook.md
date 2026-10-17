# Native GPU service runbook

1. 从指定 commit clone，设置 API_KEY / MODEL_ID / MODEL_REVISION / BENCH_PROFILE / HOST / PORT / HF_HOME。Daytona 基础环境提供 Ubuntu Linux、Python 3.12、NVIDIA 驱动、GPU 和非交互 sudo。setup 检查所选 Python 的开发头文件与 C/C++ 编译器，缺失时自行 apt 安装匹配版本；不依赖操作员手动补装。
2. `bash setup.sh` 安装私有依赖并等待就绪；`.runtime/engine.log`、`gateway.log`、`supervisor.log` 是过程诊断。不要发布进程环境或 API_KEY。
3. 只有 setup 成功并且 `/healthz` 返回 ready 才可调用。503 看引擎日志，401 检查 Bearer，429 检查队列，504 检查总 deadline。SSE error 或缺 DONE 是失败。
4. `BENCH_PROFILE=optimized bash setup.sh` 切换配置。`bash setup.sh --stop` 仅停止有匹配启动标识的本项目 supervisor；禁止 killall python。
5. 引擎关键版本在 requirements-runtime.lock；可选 SGLANG_PYTHON 选择平台固定运行时，版本不符时失败。默认则将引擎安装到项目 venv，缓存只节省下载。
6. 实验使用独占 GPU，运行前停止服务，完成后需要显式 `bash setup.sh` 恢复。保留 manifest、原始数据和失败记录。
7. 服务进程停止不代表 GPU 费用结束。使用平台结束按钮并确认 Daytona 环境销毁；正式评测由平台设置硬截止并清理。
8. 安装失败、缓存命中、模型加载和预热时间分开记录，不将缓存启动冒称为完整冷启动。
9. 正式评分由平台可信控制器独立测量；仓库中自测输出不等于正式成绩。push/CI 不申请 GPU。

历史 Lambda Docker 入口保留为 setup-lambda.sh，仅用于兼容与历史复现；当前 Daytona 优化配置需要当前硬件实测。
