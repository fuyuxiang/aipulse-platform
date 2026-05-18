# 安全设计

- 认证：用户名密码、JWT Access Token、Refresh Token、Token 吊销、API Key、Service Account 数据表。
- 密码：PBKDF2-HMAC-SHA256 加盐哈希。
- 租户隔离：业务表包含 `tenant_id`，Repository 按当前租户过滤。
- RBAC：用户、组织、角色、权限、用户角色和角色权限。
- 审计：所有 Service 写操作进入 `audit_logs`，每条记录包含 `hash` 和 `previous_hash`。
- 安全中心：敏感规则、内容安全策略、Prompt 注入规则、IP 白名单、API 限流、secret reference、风险审批策略。
- Runtime：按 tenant/agent/version/session 创建独立 workspace，工具策略和记忆策略通过桥接层注入 echo-agent。

