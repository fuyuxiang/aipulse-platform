# API 分组

OpenAPI 地址：`http://127.0.0.1:8000/docs`

主要分组：

- `/api/v1/auth/*`: 登录、刷新、吊销、登出、当前用户、菜单权限。
- `/api/v1/tenants`, `/users`, `/orgs`, `/roles`, `/permissions`: 多租户与 RBAC。
- `/api/v1/agents*`: Agent CRUD、版本、发布、灰度、回滚、调试、模板、导入导出。
- `/api/v1/model-providers`, `/model-credentials`, `/model-endpoints`, `/models`: 模型管理中心。
- `/api/v1/model-routing-policies`, `/models/route`, `/models/invoke`: 多模型调度中心。
- `/api/v1/runtime/*`: echo-agent Runtime 生命周期和 debug-run。
- `/api/v1/workflows*`: Workflow 编排、版本、发布、运行、审批和 WebSocket。
- `/api/v1/knowledge-bases*`: RAG 知识库、文档、解析、索引、检索、rerank。
- `/api/v1/tools*`, `/mcp-servers*`: 工具中心、审批、限流、MCP。
- `/api/v1/memories*`: 记忆中心。
- `/api/v1/observability/*`: metrics、logs、traces、dashboard、health。
- `/api/v1/audit-*`: 审计日志、导出、hash chain 校验。
- `/api/v1/security/*`: 敏感规则、安全检查、脱敏、密钥引用。
- `/api/v1/evaluation/*`, `/bad-cases`: 质量评测与 Bad Case。
- `/api/v1/alert-*`, `/alerts/*`: 告警规则和事件。

