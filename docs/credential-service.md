# 凭据、用量与实验账本（credential-service/1）

这是项目内真实运行的管理服务，不是平台自身用户数据库，也不是支付系统。管理与推理使用同一个网关，SQLite WAL 保存 tenant、key 摘要、不可变价格、请求账单、调整和审计。项目新增管理能力在测试环境验证中；历史 A10 与旧 Daytona 性能结果不包含这版网关的计费开销。当前 7B/V5 测量保留鉴权、配额和账本功能，状态见 [Daytona 实测记录](daytona-v5.md)。

## 启动与权限

`API_KEY`、`KEY_HASH_SECRET`（至少 32 字节）、`DATA_DIR` 从环境注入；管理端另外使用 `ADMIN_API_KEY`，不得与 `API_KEY` 相同。缺少 DATA_DIR/pepper 拒绝启动；缺少 admin key 时管理 API 关闭并发出明确提示。应用不生成公共默认密码。setup 不清空数据库；同一目录换 pepper 会拒绝启动，防止已有密钥不可用。目录 0700、主数据库 0600，原始密钥只经 HTTPS/SSH tunnel 返回一次。生成密钥包含 32 字节安全随机量，持久化 HMAC-SHA256 摘要与掩码，pepper 不落盘。

普通 API key 调用推理、`/v1/usage`；管理员凭据仅用于 `/admin/*`，不能用于推理。普通 key 没有管理员权限。`/console` 是公开登录壳，页面中没有任何租户信息或内置凭据；管理员输入的凭据只在该页内存中用于 Bearer header。刷新/退出清除，不使用 cookie，所以无需 cookie CSRF 会话；管理界面使用 textContent 渲染外部字符串，避免 HTML 注入。

`API_KEY` 被导入普通的 Benchmark 租户，给定充分实验额度（10^15 microUSD、10^12 token），照常预占/计费。它不是免计费或权限绕过入口。修改 bootstrap API_KEY 会给该持久化 tenant 增加一把普通 key，不自动撤销原 key；密钥生命周期应在管理台显式操作。

## 管理 API

成功创建 tenant/key 为 201；其他成功操作 200。错误为 `{"error":{"code":"...","message":"..."}}`，幂等冲突可增加原始 `request_id`。响应含服务端 `X-Request-ID`，时间统一 UTC ISO8601。请求模型拒绝未知字段及把 bool 当整数金额。

| 接口 | 请求 | 响应重点 |
|---|---|---|
| POST /admin/tenants | name, balance_micro_usd, token_limit, rpm_limit, concurrency_limit | tenant_id、原设置、status、余额与累计/预占用量 |
| GET /admin/tenants | 无 | items |
| PATCH /admin/tenants/{tenant_id} | status=active/disabled、token_limit、rpm_limit、concurrency_limit 任意子集 | 更新后的 tenant；不能直接写 balance/used_tokens |
| POST /admin/tenants/{tenant_id}/credits | delta_micro_usd（非零）、reason；必需 Idempotency-Key | adjustment_id、balance_micro_usd；相同请求重复返回原调整 |
| POST /admin/tenants/{tenant_id}/keys | name、expires_at（带时区或 null） | key_id、api_key（仅这次）、masked_key、status、created_at、expires_at |
| GET /admin/tenants/{tenant_id}/keys | 无 | items；不返回 raw key 或 hash |
| POST /admin/keys/{key_id}/rotate | expires_at 或 null | 新 key 信息、replaced_key_id、replaced_status=revoked |
| POST /admin/keys/{key_id}/revoke | 空 | key_id、status=revoked；重复撤销幂等 |
| GET /admin/pricing | 无 | items：不可变历史版本，最新在前 |
| PUT /admin/pricing | model、input_micro_usd_per_token、output_micro_usd_per_token | 新 pricing_version；不改旧版本 |
| GET /admin/usage | tenant_id、可选 from/to、cursor、limit（≤500） | summary、items、next_cursor；from 含、to 不含 |
| GET /admin/audit | 可选 tenant_id、cursor、limit | items、next_cursor；detail 含调整原因与恢复证据 |
| GET /v1/usage | 仅 cursor/limit | 当前 key 的 tenant summary/items；传 tenant_id/user_id 等范围覆盖参数 403 |

summary 的 `balance_micro_usd` 是尚未扣除预占的总余额，`available_micro_usd=balance-reserved`；另有 `reserved_micro_usd`、`used_tokens`、`reserved_tokens`、`input_tokens`、`output_tokens`、`cost_micro_usd`。明细含 request_id、tenant_id、key_id、parent_request_id、route、model、status、input/output/cached_tokens、cost_micro_usd、pricing_version、时间及 error_code。未知用量/费用为 null，不伪造 0。

按 tenant 选取账单后，可用每行的 key_id/model/created_at 分组，跨页加总明细与 summary 核对。`from/to` 过滤计费窗口，余额与预占仍表示当前 tenant 状态；summary 中 used_tokens 是终身 quota 计数。

## 结构化推理的 JSON 模式

`POST /v1/chat/completions` 支持标准 OpenAI 参数
`"response_format": {"type": "json_object"}`，流式和非流式都可使用。
Agent 与性能测试中的结构化任务可以显式请求这一模式，由 SGLang 约束生成 JSON
对象。当前只接受 `json_object`；其他类型、缺少 type 或附加字段均返回 422。
省略该参数或传 null 保持原有行为，不自动启用 JSON 模式。

网关原样透传此参数，不修改 messages、模型选择或 token 计数，也不去掉 Markdown
围栏、补字段、解析或重写生成内容。JSON 模式只约束输出格式，字段类型、必填内容及
语义正确性仍须由调用方严格校验；它不保证模型提取的值正确。计费仍使用实际输入计数
和引擎返回的真实 usage。内置 `/agent/run` 的模型调用也使用 `json_object` 模式。

## 预占、计费与幂等

正式服务用固定 model revision 的 tokenizer 和 `apply_chat_template(add_generation_prompt=True)` 计算真实输入 token。计数包含 system/user/assistant 文本与 chat template，不能相信客户端提交的 usage、cost、user_id（公开模型禁止这些字段）。前缀命中仍按输入计费，cached_tokens 另记；精确服务计数与引擎 usage 不一致时返回错误并走未知用量对账，不捏造费用。

一次 SQLite `BEGIN IMMEDIATE` 事务检查 tenant/key 状态、幂等、余额、累计 token、RPM 和并发，按 `输入tokens×输入价格 + max_tokens×输出价格` 预占。配额由同一用户的所有 key 共享，用户之间独立。余额/累计 token 不足 402 insufficient_quota；RPM/并发 429 并带 Retry-After；错误/撤销/到期 key 401，停用 tenant 403。租户并发与全网关 FIFO gate 是两层独立限制。

价格版本在 admission 时固定。引擎完整可信 usage 先写 journal，然后结算一次：扣实际费用、累加实际 token、释放未用预占。输入 100/输出 50，在 2/6 microUSD 价格下费用为 500。SSE 网关始终要求引擎 include_usage，客户端是否请求 usage 不影响账本；最终 usage 必须在发给客户端前落盘，收到 DONE 后正常结算。

Idempotency-Key 以 tenant+route 为范围，正文规范化哈希匹配；进行中返回 409 request_in_progress，结束后返回 409 request_already_completed 和原 request_id，不重新调用引擎；同 key 不同正文 409 idempotency_conflict。未带键的请求是独立调用。充值有独立幂等表。

本参考实现执行两次模型调用；合同允许更多调用，只要真实账目与调用次数一致。Agent 父请求占用一次 tenant RPM/并发并负责客户端幂等，父编排记录自身 0 token/费用；两次真实模型调用分别以 `parent_request_id` 关联子账单，调用前各自计数/预占，调用后分别结算。第一步完成而第二步失败时，已实际使用的第一步仍保留准确账单，不重复聚合扣费。轮换/撤销之后旧 key 不再准入新请求；已经准入的 Agent 父请求可继续完成，子账仍归属原 key。

平台 Agent 账单验收使用响应 `X-Request-ID` 定位 `route=/agent/run`、`parent_request_id=null` 的零费用父账；每次模型调用保存 `route=/agent/model_call` 的子账并关联父 request_id。`usage.model_calls` 必须为整数且至少 2，等于独立子账条数；`usage.completion_tokens` 等于子账输出 token 总数。每条子账有自己的不可变价格版本，全部归属调用者的 tenant/key/model。父账不重复扣费，同一 Idempotency-Key 重试返回 409，不能重复记账或改变另一用户账本。

## 断流与崩溃恢复

服务每个 DATA_DIR 仅允许一个网关 worker，由 OS flock 保证进程所有权。`DATA_DIR/gateway.pid` 记录当前 gateway PID；杀进程时仍必须核验进程身份/监听 socket，不能盲信文件。

取消关闭上游流并发送同一 request_id 的 abort。执行名额立即释放；账务状态 pending_reconciliation 继续保留余额/token 预占。后台每 2 秒对已结束请求处理，启动时先恢复所有孤儿请求：有完整 journal 就幂等结算；没有可信终态 usage 则记录 credit_adjustment，释放该笔预占，账单变 adjusted_unknown，input/output/cost 仍为 null。审计保存释放额度和未知用量事实。不能把未知处理称为真实零费用推理。已结算账本重复恢复不再扣费。

这份实现没有可查询历史 usage 的 SGLang API，因此对于崩溃前尚未 journal 的结果，只能采取有审计的实验额度调整，无法追溯真实已消耗的 token。SGLang abort 若失败，保留请求未知状态证据和网关 abort_failed 指标；实验额度释放不是对云厂商真实费用的退款。

## CPU 验收与真实 GPU 自测

```
python scripts/check_control_plane.py --suite mock --out results/control-plane/mock-1.json
python scripts/check_control_plane.py --base-url http://127.0.0.1:8000 --suite live --out results/control-plane/live-1.json
```

mock 通过临时目录和显式测试凭据运行真实网关/SQLite，模型用受控 CPU backend；需先安装 README 的 pytest 依赖。覆盖用户隔离、密钥生命周期、精确计费、20 并发重试/预占、跨 key 配额、真实 socket 取消和 journal/未知恢复。第 12 项真实 GPU 场景明确 skipped。live 创建两个隔离实验用户，各执行非流、流式和 Agent 工具往返；按实际账本核对归属、四条模型子账与 summary，最后撤销测试 key。live 不执行破坏性的重启或改变全局价格，1–11 需要确定性 fixture 的场景明确 skipped。

输出 `credential-service-test/1`，含 commit_sha、时间、每项 test_id/expected/actual/passed/evidence。skipped 从不标 passed；输出没有原始密钥。公开报告 `official_score=false`，独立平台评测器必须自己发请求/注入故障/扫描存储，不能信任学生提供的报告。

平台可提供独立、仅本地监听的模型 fixture：

```
ENGINE_BASE_URL=http://127.0.0.1:30001 bash setup.sh --contract-test
```

此模式只装网关依赖，使用同一真实 SQLite/鉴权/计费实现，不装模型、不启动 SGLang。fixture 的 `POST /tokenize` 接收 `{model,messages,add_generation_prompt:true}` 返回可信 `{count:100}`，推理响应提供 usage。fixture 的故障控制接口不暴露到学生网关；请求没有切换 usage/价格/故障的 test hook。`setup.sh --stop` 只停止本项目 supervisor/gateway，外部 fixture 保持，DATA_DIR 保持；正式 setup 恢复真实 SGLang 与固定 tokenizer。CPU 通过不替代 GPU 第 12 项或性能验收。
