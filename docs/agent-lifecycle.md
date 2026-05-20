# Agent Lifecycle — 配置面 vs 执行面

> AIPulse 由 **Control Plane（backend）+ Runtime Adapter（runtime/）+ Echo Agent 引擎** 三层组成。本文明确每层职责与衔接边界，避免重复实现。

---

## 1. 三层职责

| 层 | 物理位置 | 职责 | 不做什么 |
|---|---|---|---|
| **Control Plane** | `backend/app/services/build/` 等 | 资源 CRUD、版本、租户隔离、RBAC、审批、审计、配额 | 不直接执行 LLM 调用、不管理会话状态 |
| **Runtime Adapter** | `runtime/adapter/` | Control Plane 的资源（Agent/Tool/Knowledge/Memory/Model）→ Echo Agent 实例所需的 `RuntimeContext` 翻译 | 不持久化业务数据、不暴露 HTTP API |
| **Echo Agent 引擎** | `echo-agent/echo_agent/` | Agent Loop、模型调用、工具执行、记忆读写、知识检索、Trace 输出 | 不感知租户、不感知 RBAC、不直接读 Control Plane 数据库 |

**核心原则**：Control Plane 拥有"定义"（DB row），Echo Agent 拥有"运行"（in-memory instance），Runtime Adapter 是它们之间唯一的翻译器。

---

## 2. 一次会话的完整调用链

```
┌─────────────┐    ① POST /chat/completions    ┌──────────────────┐
│   Frontend  │ ────────────────────────────▶  │  api/v1/chat.py  │
└─────────────┘                                 └────────┬─────────┘
                                                          │ ② AuthService.verify
                                                          │   RBACService.allow
                                                          ▼
                                            ┌──────────────────────┐
                                            │ services/runtime/    │
                                            │ chat_service.py      │
                                            │ + agent_runner_      │
                                            │   service.py         │
                                            └──────┬───────────────┘
                                                   │ ③ load Agent / Tool / KB
                                                   │   refs from DB
                                                   ▼
                                            ┌──────────────────────┐
                                            │ runtime/adapter/     │
                                            │ config_bridge.py     │
                                            │ + tool_policy_bridge │
                                            │ + knowledge_bridge   │
                                            │ + memory_bridge      │
                                            │ + model_bridge       │
                                            └──────┬───────────────┘
                                                   │ ④ build RuntimeContext
                                                   ▼
                                            ┌──────────────────────┐
                                            │ runtime/adapter/     │
                                            │ instance_manager.py  │
                                            │ → EchoAgentRuntime   │
                                            │   Adapter.create()   │
                                            └──────┬───────────────┘
                                                   │ ⑤ in-memory Agent
                                                   │   .agent_loop.process()
                                                   ▼
                                            ┌──────────────────────┐
                                            │ echo-agent/          │
                                            │ echo_agent/agent/    │
                                            │ + models/ + skills/  │
                                            │ + memory/ + knowledge│
                                            └──────┬───────────────┘
                                                   │ ⑥ telemetry + result
                                                   ▼
                                            ┌──────────────────────┐
                                            │ runtime/adapter/     │
                                            │ telemetry_bridge.py  │
                                            │ → SQLite trace_spans │
                                            └──────────────────────┘
```

---

## 3. 资源映射表（Control Plane → Echo Agent）

| Control Plane 资源 | DB 表 | 桥接文件 | Echo Agent 概念 |
|---|---|---|---|
| Agent definition | `agents` | `config_bridge.py` | `AgentConfig` |
| Tool registration | `tools` | `tool_policy_bridge.py` | `Skill` + `PermissionPolicy` |
| Knowledge base | `knowledge_bases` + `documents` | `knowledge_bridge.py` | `KnowledgeStore` retriever |
| Memory scope | `memory_items` + `memory_scopes` | `memory_bridge.py` | 4-layer memory writers |
| Model deployment | `model_deployments` + `model_routing_policies` | `model_bridge.py` | `ModelClient` (provider + key) |
| Session | `chat_sessions` + `chat_messages` | `session_bridge.py` | `SessionKey` |
| Trace span | `trace_spans` | `telemetry_bridge.py` | OpenTelemetry exporter |

---

## 4. 谁负责什么 — 反例查找指南

遇到不确定该写在哪层时，按下表对照：

| 需求 | 应该写在 | 不应写在 |
|---|---|---|
| 给 Agent 加一个新版本 | `services/build/agent_service.py` | echo-agent |
| 给 Agent 加一种新工具调用方式 | `echo-agent/echo_agent/skills/` | backend |
| 检查用户是否能调用某 Agent | `services/_shared/auth_service.py` + `rbac_service.py` | echo-agent |
| Agent 决定是否调用某工具 | `echo-agent/echo_agent/agent/` (LLM 决策) + `permissions/` (策略放行) | backend |
| 记录一次会话花了多少 token | `services/observe/cost_analytics_service.py`（消费 telemetry） | echo-agent |
| 计算 token 用量 | echo-agent telemetry | backend |
| 让某租户每天最多 1000 次调用 | `services/settings/` 配额配置 + `services/_shared/auth_service.py` 拦截 | echo-agent |
| 工具调用本身的限流 / 重试 | `runtime/adapter/resource_limits.py` | backend |

---

## 5. 数据双向流动

**Control Plane → Echo Agent（启动时）**：
1. 用户调用 `/chat/completions` 触发 Agent 实例化
2. `agent_runner_service` 从 DB 读取 Agent 定义、引用的工具/知识库/记忆/护栏/路由策略
3. `config_bridge` 把它们组装成 `RuntimeContext`
4. `instance_manager` 在 `data/runtime/<instance_id>/` 准备工作区，启动 Echo Agent 进程内实例

**Echo Agent → Control Plane（运行时）**：
- Telemetry：每个 span/event 通过 `telemetry_bridge` 写入 `trace_spans` 表
- Memory：通过 `memory_bridge` 写入 `memory_items`（按 scope 隔离）
- Session：消息通过 `session_bridge` 写入 `chat_messages`

**注意**：Echo Agent **不读** Control Plane 的 DB。所有上下文必须在启动时完整下发，运行时只能写出（telemetry / memory / session）。这保证了：
- 引擎可独立部署、独立测试
- Control Plane 的鉴权策略一定在调用前生效，不会被引擎绕过

---

## 6. 边界故障排查

| 现象 | 可能原因 | 检查点 |
|---|---|---|
| Agent 跑起来但用错了模型 | model_bridge 没把路由策略翻译对 | `runtime/adapter/model_bridge.py` |
| 工具调用被拦截但策略允许 | tool_policy_bridge 同步延迟 | 重启实例（`instance_manager.restart`）|
| 知识检索召回为空 | knowledge_bridge 给的是空 retriever | 检查 KB 是否完成索引（`knowledge_bases.status == 'ready'`）|
| 看到执行成功但 trace 缺 span | telemetry_bridge 异步写失败 | `data/traces/backend-traces.jsonl` |
| 同一 Agent 不同租户互相串了上下文 | session_bridge 的 session_key 没带 tenant_id | `runtime/adapter/session_bridge.py:session_key()` |

---

## 7. 不变式（修改时务必保留）

1. **Echo Agent 不引用 backend.app.\*** — 引擎是独立 Python 包，可单独 pip install
2. **backend 不直接 import echo_agent.agent.\*** — 必须经 `runtime.adapter.\*`
3. **每个跨层调用必经 RuntimeContext** — 不允许 Echo Agent 直接读 Request / Session
4. **租户上下文一定在 RuntimeContext 里** — 不允许在 Echo Agent 内部"猜"
5. **配置面是 source of truth** — DB 改动后，运行中的实例需 restart 才生效（`instance_manager.restart`），引擎不做热配置更新
