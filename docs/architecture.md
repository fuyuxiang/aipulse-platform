# AIPulse 本地调试版架构

## 目标架构

AIPulse 使用 Control Plane + Runtime Plane 双层架构。当前阶段以本地源码和本地进程运行，不交付 Docker、Kubernetes、Helm、Prometheus/Grafana/ELK/Jaeger 部署配置。

## 目录结构

- `echo-agent/`: 底层 Agent 执行引擎，保留 Agent Loop、工具、记忆、模型路由、MCP、A2A、Gateway、可观测与安全审批能力。
- `backend/`: FastAPI 企业控制平面，提供租户、RBAC、Agent、模型、Workflow、RAG、工具、记忆、审计、安全、评测、告警和 Runtime API。
- `runtime/`: echo-agent 企业运行时适配层，负责实例隔离、生命周期、任务执行、Workflow 执行、本地存储、Telemetry 和桥接。
- `frontend/`: React + TypeScript + Ant Design Pro 中台。
- `scripts/`: 本地初始化、启动、测试和清理脚本。
- `data/`: SQLite、文件、向量索引、日志、Trace 和导出目录。
- `docs/`: 架构、接口、规范、安全、测试和本地运维文档。

## 模块边界

Control Plane 只管理企业级元数据、权限、审计、策略和可视化配置，不把平台代码塞入 `echo-agent/echo_agent`。Runtime Plane 通过 `runtime/adapter/echo_agent_adapter.py` 创建 echo-agent 实例，并注入 tenant、agent、version、session、模型、工具、记忆、知识库、安全和资源限制配置。

## 数据流

前端请求后端 API，后端鉴权和租户上下文解析后进入 Service/Repository。写操作记录 `audit_logs` hash chain。Runtime 请求进入 `RuntimeControlService`，再调用 `EchoAgentRuntimeAdapter` 创建或运行 echo-agent 实例。模型调用、工具调用、RAG 检索、Workflow 执行和评测结果写入本地 SQLite 与 JSONL/Trace 文件。

## echo-agent 审计结论

当前 `echo-agent` 可复用模块：

- `echo_agent.agent.loop.AgentLoop`: Agent Loop 和 pipeline。
- `echo_agent.gateway.server.GatewayServer`: 原生 Gateway 参考，不作为企业控制平面。
- `echo_agent.memory.*`: 原生记忆引擎，由平台记忆中心桥接。
- `echo_agent.models.*`: Provider、路由、retry、tokenizer，由平台模型中心和路由中心治理后注入。
- `echo_agent.agent.tools.*`: 内置工具和工具注册表，由平台工具中心做权限、审批和审计。
- `echo_agent.security.*`: 工具安全、路径策略和智能审批能力。
- `echo_agent.observability.*`: OpenTelemetry 与 trace logger。
- `echo_agent.a2a.*`、`echo_agent.mcp.*`: A2A 与 MCP 能力。
- `echo_agent.tasks.*`、`echo_agent.knowledge.*`: 任务/Workflow 与知识索引底层能力。

当前不需要修改 `./echo-agent`。企业平台通过 editable install、`PYTHONPATH` 和 runtime bridge 引用原引擎，避免复制和嵌套。

## 未来生产替换点

- SQLite -> MySQL/PostgreSQL。
- LocalTaskExecutor -> Celery/MQ。
- LocalVectorStore -> Milvus。
- 本地文件系统 -> MinIO。
- JSONL 日志 -> ELK。
- 本地 Trace -> Jaeger。
- metrics API -> Prometheus。
- 本地配置 -> 配置中心。
- 单机 Runtime Plane -> 分布式执行集群。

