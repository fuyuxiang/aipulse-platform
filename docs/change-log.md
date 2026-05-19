# Change Log

## 2026-05-18

新增本地调试版企业级智能体平台目录：

- `backend/`
- `runtime/`
- `frontend/`
- `scripts/`
- `data/`
- `docs/`

`./echo-agent` 未做源码修改。

### echo-agent 修改说明

- 修改文件：无。
- 修改原因：不需要修改，当前通过 `runtime/adapter` 和 editable install/PYTHONPATH 引用。
- 影响范围：无原引擎侵入。
- 回滚方式：删除新增平台目录即可恢复原 `echo-agent` 状态。
- 测试覆盖：runtime adapter 生命周期测试、debug-run 测试、后端 Runtime API 集成测试。

## 2026-05-19

补强本地调试版企业平台的专用业务能力：

- 新增 `backend/app/services/knowledge_service.py`：文档本地存储、解析、分块、embedding、LocalVectorStore 写入、检索、rerank、删除向量、索引统计。
- 新增 `backend/app/services/workflow_service.py`：DAG 校验、版本快照、发布、执行、节点步骤日志、人工审批暂停/决策、重试、取消、回放。
- 新增 `backend/app/services/tool_service.py`：工具 JSON Schema 参数校验、风险审批、限流、调用执行、调用日志、MCP 工具同步。
- 新增 `backend/app/services/memory_service.py`：记忆抽取、检索、合并、归档、脱敏、过期清理、冲突解决、记忆审计。
- 增强 `backend/app/services/model_services.py`：fixed/weighted/priority/cost/latency/quality 路由、能力过滤、quota、circuit breaker、credential/endpoint rotation 选择。
- 增强 `backend/app/services/resource_service.py`：观测写入后按本地告警规则触发 `alert_events`。
- 新增 `backend/app/services/agent_service.py`：Agent clone、版本快照、发布、灰度、回滚、导入导出、调试运行、运行状态聚合。
- 增强 `backend/app/services/audit_service.py`：审计日志 JSONL 文件导出和导出文件 sha256。
- 增强 `backend/app/services/security_service.py`：敏感规则、内容安全策略、Prompt 注入规则、IP allowlist、本地 API 限流规则统一检查；`secret_refs` 和 `model_credentials` 写入时做密钥引用和脱敏。
- 增强 `backend/app/services/evaluation_service.py`：评测结果逐 case 落表、Prompt 对比、回归评测。
- 增强 `backend/app/services/knowledge_service.py`：支持 txt、markdown、csv、html、docx、xlsx、pdf 的本地解析，支持 base64 上传。
- 增强模型管理路由：Provider capabilities 自动声明、Provider 下凭证创建、凭证可用性测试、模型版本、模型健康检查、熔断重置。
- 增强前端 `frontend/src/components/FeatureWorkbench.tsx`：在相关页面直接调用真实接口执行知识库、Workflow、工具、记忆、模型、模型路由、安全、评测操作。
- 新增业务断言测试 `backend/tests/integration/test_domain_behaviors.py` 和 `frontend/tests/unit/featureWorkbench.test.ts`，覆盖 Agent 发布链路、审计导出、安全策略、密钥脱敏、评测结果、告警触发、RAG、Workflow、工具审批、记忆、模型路由。

`./echo-agent` 未做源码修改。
