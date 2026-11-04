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


## 2026-11-04 评审回归

- `backend/tests/integration/test_review_regressions.py`：普通用户授撤权、租户与会话隔离、审批防伪/绑定/到期/单次消费、模型故障、真实下游工具效果、进程重启检查点恢复、取消传播、发布快照及 Key、记忆范围、导入答案评分。
- `runtime/tests/test_policy_bridges.py`：工具 allow/deny、禁用记忆、未知 Provider 禁止静默演示。
- `frontend/tests/e2e/business.spec.ts`：隔离后台和浏览器，从正常菜单操作用户/聊天/Agent 向导/工作流/团队；使用本地演示 Provider，不作为真实模型质量证据。
- `npm run e2e` 自动创建临时 SQLite/文件/向量目录，在 127.0.0.1:3100 和 :8107 启动测试服务；端口应空闲。需要安装与 Playwright 对应的 Chromium，或用 `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` 指定可用本机浏览器。证据位置会打印在日志中。
- `scripts/run_tests.py` 是 pytest/unit 汇总，不含 lint、build、E2E 和真实模型验收，不能据其成功宣称完成全部验收。

真实模型验收为显式 opt-in，使用独立临时数据库，仅发送合成数据：

```bash
PYTHONPATH=backend python3 scripts/review_real_model_smoke.py
# 使用 LLM_BASE_URL / LLM_MODEL / LLM_API_KEY；不打印密钥。
# 已授权且不需鉴权的测试网关：
PYTHONPATH=backend python3 scripts/review_real_model_smoke.py --base-url "$REVIEW_BASE_URL" --model 'Echo Cedar' --tool-model 'Echo Reed' --no-auth --extended
```

extended 模式假设同一测试网关提供 `embed-pro`、`rerank-pro` 和支持函数调用的 tool-model，验证真实文档/查询向量、重排、工具副作用日志和三成员流水线。若模型不支持函数调用或未返回所需调用，验收应失败，不能用回答中“已调用工具”的文字替代工具日志。

## 前端产品化回归（2026-11-04）

- `frontend/tests/e2e/product.spec.ts`：列表首屏层级、跨页服务端搜索、编辑/启停/删除持久化、桌面搜索与手机导航、26 个导航页面的脚本及 API 错误、API 发布与停用、工作流未保存保护及审批后继续、模型连接表单、知识库配置/文档上传/检索。
- `frontend/tests/unit/access.test.ts`：只读角色、领域通配权限、租户管理边界。
- `frontend/tests/unit/markdown.test.tsx`：表格与代码展示，不执行 HTML、不自动加载模型回答中的外部图片。
- 后端新增回归：编辑遮罩配置不覆盖服务端密钥、团队编辑在引用无效时不部分保存、版本列表过滤及跨智能体版本发布拒绝。
- Vitest 使用 `vitest.config.mts`；Node.js 要求见 `frontend/package.json`。依赖升级后必须一起运行类型检查、单元测试、构建和浏览器回归。
