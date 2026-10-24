# 原生运行时复现边界

## 下载缓存与首次安装

Daytona 提供的 `/opt/llm-cache/wheels` 只是下载缓存，不是已经装好的项目环境。仅设置 `PIP_FIND_LINKS` 不能保证 pip 优先使用本地 wheel；2026-10-24 实测发现它仍会选择公网的同版本 PyTorch，重复下载约 900 MB。

现在默认原生 setup 在目标 venv 的 Python 下运行 `scripts/wheel_cache.py`，只选择完整锁中版本完全匹配、且 Python/ABI/平台标签兼容的本地 wheel，先用 `--no-index --no-deps` 安装。随后正常执行完整依赖解析、`pip check` 和 175 包版本集合核对；缺少的包从原配置的索引补齐。不安装缓存中的其他版本或锁外包；不改变模型、精度、性能门槛或安装时限。

缓存可选：未提供 `PIP_FIND_LINKS` 或没有匹配 wheel 时走正常网络安装。网络冷下载和命中下载缓存的耗时必须分别说明；两者仍从新项目 venv 安装。损坏的已选缓存会明确报错，不静默跳过校验。CPU `--contract-test` 和显式外部引擎模式不会预装 GPU 依赖。

## 锁定与复现

`requirements.lock` 是网关与 CPU 测试依赖，`requirements-runtime.lock` 是引擎关键约束，`requirements-analysis.lock` 是实验与绘图依赖，`requirements-tokenizer.lock` 固定 tokenizer 依赖。`requirements-native-linux-py312.lock` 补全 Linux x86_64 / Python 3.12 的所有已安装发行包版本。默认原生 setup 在一次 pip 解析中同时应用这五份文件，随后运行 `pip check` 并核对完整集合；不会让第二次引擎安装悄悄升级先前固定的网关依赖。

完整锁来自 Daytona 运行 `39065b3b-fa51-4078-8b5c-ea322efa9fa8`、源码 `ce1fa35c2592381a201b5ccf4f2a599d10a8c93b` 的 175 个发行包，Python 3.12.3，pip check 为 0。捕获时依赖安装成功；这本身不代表整套 GPU 评测通过。安装后实际 idna 为 3.20，因此网关约束同步为 3.20，避免与实际兼容集合冲突。捕获记录不含 direct URL、本地路径安装或凭据。

`.runtime/native-dependencies.json` 保存实际名称/版本集合、数量和锁文件 SHA256；`.runtime/runtime.json` 保存是否应用完整锁及其 SHA256。所有依赖文件都进入 setup 的幂等 fingerprint，修改分析依赖也会触发重新安装和启动。若项目 venv 留有锁外软件或版本漂移，setup 明确失败；清理本项目 venv 后重建，不修改系统 Python。

此锁固定 Python 包版本，不等于 wheel 哈希锁，也不固定 NVIDIA 驱动、CUDA toolkit、Ubuntu apt 包或模型文件。基础快照、Python/CUDA/驱动版本、模型 revision 与源代码 commit 仍须记录在每次评测报告中。模型版本和服务精度不会因导出依赖锁改变。

可选 `SGLANG_PYTHON` 路径用于显式选用外部预置引擎：网关仍私有安装，外部引擎执行关键版本和 pip check，但必须标明 `native_lock_applied=false`，不能冒称完整私有环境重建。正式独立确认使用默认私有环境路径。

CPU CI 只安装网关与测试依赖，不安装 Linux GPU 闭包，不申请 GPU。完整锁的后续独立重建已由 [Daytona V5 两轮实测](daytona-v5.md) 验证；原始依赖捕获本身不等于通过，不能用同一台已有环境的检查替代独立重建。
