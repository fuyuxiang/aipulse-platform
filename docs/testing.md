# 测试

后端：

```bash
cd backend
ruff check .
mypy app
pytest
```

Runtime：

```bash
cd runtime
pytest
```

前端：

```bash
cd frontend
npm run lint
npm run test
npm run build
npm run e2e
```

测试覆盖：

- 登录、JWT、RBAC 拒绝、审计 hash chain。
- Agent、模型、知识库、记忆、安全、评测 API。
- OpenAPI 契约关键路径。
- Runtime adapter 生命周期和 echo-agent debug run。
- Workflow DAG 校验、环检测和节点执行。
- LocalVectorStore 向量、关键词、混合检索和删除。
- 前端页面配置、服务层和登录页 E2E。

